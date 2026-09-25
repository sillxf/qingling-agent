from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import Settings, settings
from .event_source import DemoEventSource, EventSource, EventRequestContext, build_event_source
from .model_gateway import ModelCallContext
from .models import ToolContract
from .rag import KnowledgeAccessContext, RAGService


@dataclass(frozen=True)
class ToolInvocationContext(EventRequestContext):
    """Trusted execution context supplied by RuntimeService to a tool."""

    run_id: str
    tenant_id: str
    user_id: str
    correlation_id: str
    roles: List[str] = field(default_factory=list)
    deadline_monotonic: Optional[float] = None
    cancel_event: Optional[threading.Event] = None

    def remaining_seconds(self, default: float) -> float:
        if self.deadline_monotonic is None:
            return float(default)
        return min(float(default), max(0.0, self.deadline_monotonic - time.monotonic()))

    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()


ToolHandler = Callable[..., Dict[str, Any]]


class ToolNotFound(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, Tuple[ToolContract, ToolHandler]] = {}

    def register(self, contract: ToolContract, handler: ToolHandler) -> None:
        if contract.name in self._items:
            raise ValueError(f"tool already registered: {contract.name}")
        self._items[contract.name] = (contract, self._adapt_handler(handler))

    @staticmethod
    def _adapt_handler(handler: ToolHandler) -> ToolHandler:
        """Keep compatibility with existing one-argument tool handlers."""

        try:
            signature = inspect.signature(handler)
            try:
                signature.bind({}, None)
                return handler
            except TypeError:
                signature.bind({})
        except (TypeError, ValueError):
            return handler

        def legacy_adapter(args: Dict[str, Any], _context: Optional[ToolInvocationContext]) -> Dict[str, Any]:
            return handler(args)

        return legacy_adapter

    def get(self, name: str) -> Tuple[ToolContract, ToolHandler]:
        try:
            return self._items[name]
        except KeyError as exc:
            raise ToolNotFound(f"tool not found: {name}") from exc

    def list_contracts(self) -> List[ToolContract]:
        return [contract for contract, _ in self._items.values()]

    def invoke(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        context: Optional[ToolInvocationContext] = None,
    ) -> Dict[str, Any]:
        _, handler = self.get(name)
        return handler(args, context)


def build_default_tools(
    app_settings: Settings = settings,
    *,
    event_source: Optional[EventSource] = None,
    rag_service: Optional[RAGService] = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    events = event_source or build_event_source(app_settings)

    def search_events(args: Dict[str, Any], context: Optional[ToolInvocationContext]) -> Dict[str, Any]:
        if context is None:
            if not isinstance(events, DemoEventSource):
                raise ValueError("event.search requires a trusted tool context")
            # Preserve direct offline-tool calls for existing local clients;
            # a configured remote source always requires Runtime context.
            context = ToolInvocationContext(
                run_id="local",
                tenant_id="local",
                user_id="local",
                correlation_id="local",
            )
        return events.search(query=args["query"], context=context)

    def get_event(args: Dict[str, Any], context: Optional[ToolInvocationContext]) -> Dict[str, Any]:
        if context is None:
            if not isinstance(events, DemoEventSource):
                raise ValueError("event.get requires a trusted tool context")
            context = ToolInvocationContext(
                run_id="local",
                tenant_id="local",
                user_id="local",
                correlation_id="local",
            )
        return events.get(event_id=args["event_id"], context=context)

    def search_knowledge(args: Dict[str, Any], context: Optional[ToolInvocationContext]) -> Dict[str, Any]:
        if rag_service is None:
            raise ValueError("knowledge.search is not configured")
        if context is None:
            raise ValueError("knowledge.search requires a trusted tool context")
        result = rag_service.search(
            args["query"],
            context=KnowledgeAccessContext(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                # Tool context will carry trusted roles when Runtime adds them;
                # the current contract remains least-privilege by default.
                roles=list(getattr(context, "roles", []) or []),
            ),
            top_k=args.get("top_k", 5),
            min_score=args.get("min_score", 0.0),
            include_flagged=bool(args.get("include_flagged", False)),
            embedding_context=ModelCallContext(
                deadline_monotonic=context.deadline_monotonic,
                cancel_event=context.cancel_event,
            ),
        )
        return result.model_dump() if hasattr(result, "model_dump") else result.dict()

    registry.register(
        ToolContract(
            name="event.search",
            description="Search normalized security events.",
            permission="read",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
            max_retries=2,
            retry_policy="transient",
        ),
        search_events,
    )
    registry.register(
        ToolContract(
            name="event.get",
            description="Get one event by identifier.",
            permission="read",
            input_schema={
                "type": "object",
                "required": ["event_id"],
                "properties": {"event_id": {"type": "string"}},
            },
            max_retries=2,
            retry_policy="transient",
        ),
        get_event,
    )
    if rag_service is not None:
        registry.register(
            ToolContract(
                name="knowledge.search",
                description="Search tenant-scoped knowledge and return source citations.",
                permission="read",
                input_schema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "include_flagged": {"type": "boolean"},
                    },
                },
                max_retries=2,
                retry_policy="transient",
            ),
            search_knowledge,
        )
    registry.register(
        ToolContract(
            name="report.preview",
            description="Render a report preview from structured findings.",
            permission="analyze",
            input_schema={
                "type": "object",
                "required": ["title", "content"],
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        ),
        lambda args, _context: {
            "format": "markdown",
            "title": args["title"],
            "content": args["content"],
        },
    )
    registry.register(
        ToolContract(
            name="response.block_ip",
            description="Block an IP through a security control plane.",
            permission="mutate",
            risk_level="critical",
            reversible=True,
            requires_approval=True,
            input_schema={
                "type": "object",
                "required": ["ip", "reason"],
                "properties": {
                    "ip": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        ),
        lambda args, _context: {
            "dry_run": True,
            "action": "block_ip",
            "ip": args["ip"],
            "reason": args["reason"],
            "message": "高风险处置仅生成影响预览，实际执行需要接入审批后的控制器。",
        },
    )
    return registry
