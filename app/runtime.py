from __future__ import annotations

import json
import threading
import time
import random
import uuid
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NoReturn, Optional

from .config import Settings, settings
from .decision import DecisionService, model_to_dict
from .event_source import EventSourceError
from .model_gateway import ModelCallContext, ModelGateway, ModelGatewayError, build_model_gateway
from .models import AgentProfile, Approval, Checkpoint, ComponentManifest, ComponentExecution, Compensation, DeadLetter, Run, RunCreateRequest, ToolContract, WorkflowManifest, utc_now
from .opencode import FileSystemSandbox, HookRegistry, MCPRegistry, OpenCodeResourceError, SkillRegistry, SubAgentRegistry, SubAgentDefinition, build_mcp_registry, mcp_contract
from .observability import normalize_correlation_id, redact_text, redact_value
from .policy import PolicyGate
from .rag import (
    KnowledgeAccessContext,
    KnowledgeAccessDenied,
    KnowledgeEmbeddingError,
    KnowledgeNotFoundError,
    KnowledgeValidationError,
    RAGService,
    build_knowledge_store,
)
from .state import InvalidTransition, TERMINAL_STATES, transition
from .store import NotFoundError, StoreProtocol, build_store
from .tools import ToolInvocationContext, ToolNotFound, ToolRegistry, build_default_tools
from .component_registry import ComponentContext, ComponentRegistry
from .workflow_validation import validate_workflow
from .runtime_errors import (
    RuntimeErrorBase, ApprovalRequired, PolicyDenied, RunTimeout,
    RunCancelled, TenantAccessDenied,
)
from .react_executor import ReactExecutor
from .prompt_chain_executor import execute_prompt_chain
from .builtin_components import register_builtin_components
from .workflow_catalog import WorkflowCatalog
from .workflow_executor import WorkflowExecutor
from .security_assessment import SecurityAssessmentService, register_security_assessment_component
from .security_sources import InMemoryAssetSource, InMemoryExposureSource, InMemoryVulnerabilitySource


class RuntimeService:
    def __init__(
        self,
        *,
        store: Optional[StoreProtocol] = None,
        policy_gate: Optional[PolicyGate] = None,
        tool_registry: Optional[ToolRegistry] = None,
        model_gateway: Optional[ModelGateway] = None,
        rag_service: Optional[RAGService] = None,
        component_registry: Optional[ComponentRegistry] = None,
        skill_registry: Optional[SkillRegistry] = None,
        mcp_registry: Optional[MCPRegistry] = None,
        hook_registry: Optional[HookRegistry] = None,
        sandbox: Optional[FileSystemSandbox] = None,
        subagent_registry: Optional[SubAgentRegistry] = None,
        app_settings: Settings = settings,
    ) -> None:
        self.store = store if store is not None else build_store(app_settings)
        self.policy_gate = policy_gate or PolicyGate()
        self.models = model_gateway if model_gateway is not None else build_model_gateway(app_settings)
        self.settings = app_settings
        self.components = component_registry or ComponentRegistry()
        self._register_builtin_components()
        self.security_assessment = SecurityAssessmentService(
            InMemoryAssetSource(), InMemoryVulnerabilitySource(), InMemoryExposureSource()
        )
        configured_skills = getattr(app_settings, "skills_root", "")
        skills_root = (Path(configured_skills).expanduser().resolve() if configured_skills
                       else Path(__file__).resolve().parent / "resources" / "skills")
        self.skills = skill_registry or SkillRegistry(skills_root, max_chars=int(getattr(app_settings, "skill_max_chars", 120_000)))
        self.mcp = mcp_registry or build_mcp_registry(app_settings)
        sandbox_root = getattr(app_settings, "opencode_sandbox_root", "data/opencode-sandbox")
        if not str(sandbox_root).startswith(("/", "\\")) and ":" not in str(sandbox_root):
            sandbox_root = (Path.cwd() / str(sandbox_root)).resolve()
        self.sandbox = sandbox or FileSystemSandbox(
            sandbox_root,
            max_file_bytes=int(getattr(app_settings, "opencode_max_file_bytes", 1_000_000)),
            max_output_chars=int(getattr(app_settings, "opencode_max_output_chars", 30_000)),
        )
        self.hooks = hook_registry or HookRegistry()
        self.subagents = subagent_registry or SubAgentRegistry()
        self._register_builtin_subagents()
        knowledge_backend = getattr(app_settings, "knowledge_store_backend", None)
        # Keep the knowledge store aligned with the primary persistence
        # setting when callers construct ``Settings(store_backend="sqlite")``
        # without repeating the knowledge-specific option.
        if not knowledge_backend:
            knowledge_backend = getattr(app_settings, "store_backend", "memory")
        knowledge_path = getattr(app_settings, "knowledge_sqlite_path", None)
        if not knowledge_path:
            # A shared SQLite path is the least surprising local setup when
            # the caller explicitly selects the SQLite primary store.  Keep a
            # dedicated path for the default in-memory configuration.
            knowledge_path = (
                getattr(app_settings, "sqlite_path", "data/qingling.db")
                if str(knowledge_backend).strip().lower() == "sqlite"
                else "data/qingling-knowledge.db"
            )
        self.rag = rag_service if rag_service is not None else RAGService(
            store=build_knowledge_store(
                backend=knowledge_backend,
                path=knowledge_path,
            ),
            model_gateway=self.models,
            embedding_model=getattr(app_settings, "knowledge_embedding_model", "bge-m3"),
        )
        self.tools = tool_registry if tool_registry is not None else build_default_tools(
            app_settings,
            rag_service=self.rag,
        )
        self._register_mcp_tools()
        self._register_opencode_tools()
        from .workflow_components import register_workbench_components
        register_workbench_components(
            self.components, self.tools,
            invoke_tool=self.invoke_tool,
            model_chat=self.model_chat,
            run_subagent=self.run_subagent_tool,
        )
        register_security_assessment_component(self.components, self.security_assessment)
        self.workflow_catalog = WorkflowCatalog(self.store, self.components)
        self.decision = DecisionService(
            tool_registry=self.tools,
            policy_gate=self.policy_gate,
            model_gateway=self.models,
            model_planning_enabled=bool(getattr(app_settings, "decision_model_planning", False)),
            max_steps=int(getattr(app_settings, "decision_max_steps", 12)),
        )
        self._executor = ThreadPoolExecutor(max_workers=max(1, app_settings.run_workers))
        self._cancel_events: Dict[str, threading.Event] = {}
        self._futures: Dict[str, Future[None]] = {}
        self._deadline_monotonic: Dict[str, float] = {}
        self._timeout_timers: Dict[str, threading.Timer] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._recover_persisted_runs()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for event in self._cancel_events.values():
                event.set()
            for timer in self._timeout_timers.values():
                timer.cancel()
            self._timeout_timers.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        close_store = getattr(self.store, "close", None)
        if callable(close_store):
            close_store()
        rag_store = getattr(getattr(self, "rag", None), "store", None)
        if rag_store is not self.store:
            close_rag_store = getattr(rag_store, "close", None)
            if callable(close_rag_store):
                close_rag_store()

    def publish_profile(self, profile: AgentProfile) -> AgentProfile:
        if not profile.id.strip():
            raise ValueError("agent profile id cannot be empty")
        if profile.max_react_steps < 1 or profile.max_react_steps > 100:
            raise ValueError("max_react_steps must be between 1 and 100")
        if profile.max_retry < 0 or profile.max_retry > 10:
            raise ValueError("max_retry must be between 0 and 10")
        if len(profile.few_shots) < 3:
            raise ValueError("agent profile requires at least 3 few-shot examples")
        return self.store.register_profile(profile)

    def publish_workflow(self, workflow: WorkflowManifest) -> WorkflowManifest:
        return self.workflow_catalog.publish(workflow)

    def validate_workflow(self, workflow: WorkflowManifest):
        return validate_workflow(workflow, self.components)

    def submit(self, request: RunCreateRequest, *, workflow: Optional[WorkflowManifest] = None) -> Run:
        with self._lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            profile = self.store.get_profile(request.agent_id)
            if not profile.enabled:
                raise ValueError(f"agent profile is disabled: {profile.id}")
            existing = self.store.find_idempotent_run(request.tenant_id, request.idempotency_key)
            if existing:
                # Idempotent submission is also a read of the original Run.
                # Use the same ownership rule as GET before returning its data.
                self._assert_run_access(existing, tenant_id=request.tenant_id,
                                        user_id=request.user_id, roles=request.roles)
                self._ensure_runtime_tracking(existing)
                return existing
            correlation_id = normalize_correlation_id(request.correlation_id)
            timeout_seconds = max(0.001, float(self.settings.run_timeout_seconds))
            deadline_at = utc_now() + timedelta(seconds=timeout_seconds)
            decision = self.decision.decide(request.input, profile)
            if workflow is not None:
                self.workflow_catalog.ensure_enabled(workflow.id)
                self.workflow_catalog.verify(workflow)
                if workflow.tenant_id and workflow.tenant_id != request.tenant_id:
                    raise ValueError("workflow tenant is not authorized")
                decision.route.engine = "workflow"
                decision.route.workflow_ref = workflow.id
                decision.route.workflow_version = workflow.version
                decision.intent.needs_clarification = False
                decision.plan = self.decision.plan_for_workflow(decision.intent, workflow)
                decision.validation = self.decision.validate_plan(decision.plan, profile)
                decision.plan_hash = self.decision._plan_hash(decision.plan)
            if decision.intent.task == "tool_action" and decision.route.engine != "opencode":
                decision.validation.valid = False
                decision.validation.errors.append("tool_action requires the opencode tool execution engine")
            run = self.store.create_run(request, profile, correlation_id=correlation_id, deadline_at=deadline_at)
            run.decision_snapshot = model_to_dict(decision)
            run.execution_engine = decision.route.engine
            run.workflow_ref = decision.route.workflow_ref
            run.execution_state = {
                "plan_id": decision.plan.plan_id,
                "plan_hash": decision.plan_hash,
                "status": "queued",
                "steps": {},
            }
            # Lock the workflow version at submission time so later config
            # changes do not silently alter an in-flight Run.
            if decision.route.engine == "workflow":
                workflow_id = decision.route.workflow_ref or profile.workflow_ref or profile.id
                try:
                    workflow = workflow or self.workflow_catalog.resolve(workflow_id)
                    if workflow.tenant_id and workflow.tenant_id != request.tenant_id:
                        raise ValueError("workflow tenant is not authorized")
                    run.workflow_ref = workflow.id
                    run.workflow_version = workflow.version
                    decision.route.workflow_ref = workflow.id
                    decision.route.workflow_version = workflow.version
                    # Use the actual published manifest as the plan source so
                    # custom workflows do not get an unrelated fixed template.
                    if decision.intent.task == "event_investigation":
                        decision.plan = self.decision.plan_for_workflow(decision.intent, workflow)
                        decision.validation = self.decision.validate_plan(decision.plan, profile)
                        decision.plan_hash = self.decision._plan_hash(decision.plan)
                    run.decision_snapshot = model_to_dict(decision)
                except NotFoundError:
                    # Preserve the existing failure path if the workflow is
                    # missing; execute() will surface a stable validation error.
                    pass
            run = self.store.update_run(run)
            self._ensure_runtime_tracking(run)
            # Keep the original first event for existing consumers and
            # append decision events after it.
            self.emit(
                run.id,
                "run.created",
                {
                    "agent_id": profile.id,
                    "agent_version": profile.version,
                    "engine": run.execution_engine,
                },
            )
            self.store.enqueue_run(run.id)
            self._futures[run.id] = self._executor.submit(self._process_queued_run, run.id)
            return self.store.get_run(run.id)

    def _process_queued_run(self, run_id: str) -> None:
        claimed = self.store.claim_run(f"worker-{threading.get_ident()}", run_id)
        if claimed != run_id:
            return
        try:
            self.execute(claimed)
        finally:
            current = self.store.get_run(claimed)
            if current.status in TERMINAL_STATES:
                self.store.ack_run(claimed)

    def execute(self, run_id: str) -> None:
        try:
            run = self.transition(run_id, "running")
            if run.status != "running":
                return
            self.check_runtime_limits(run_id)
            profile = self.store.get_profile(run.agent_id, run.agent_version)
            snapshot = run.decision_snapshot or {}
            self._emit_decision_events(run_id, snapshot)
            validation = snapshot.get("validation") if isinstance(snapshot, dict) else None
            engine = run.execution_engine or profile.execute_engine
            # Legacy explicit tool_call inputs keep their original fail-closed
            # ReAct/Policy error semantics; the executor still validates the
            # tool and arguments before any invocation.
            defer_explicit_tool_validation = engine == "opencode" and "tool_call" in run.input
            if isinstance(validation, dict) and not validation.get("valid", True) and not defer_explicit_tool_validation:
                errors = validation.get("errors") or ["decision plan is invalid"]
                raise ValueError("plan invalid: " + "; ".join(str(error) for error in errors[:3]))
            self.emit(run_id, "run.started", {"engine": "decision" if isinstance(snapshot, dict) and isinstance(snapshot.get("intent"), dict) and snapshot["intent"].get("needs_clarification") else engine})
            intent_data = snapshot.get("intent") if isinstance(snapshot, dict) else None
            if isinstance(intent_data, dict) and intent_data.get("needs_clarification"):
                self.step_started(run.id, "clarify", "clarify")
                self.step_completed(run.id, "clarify", "clarify", output_keys=["message"])
                output = {
                    "engine": "decision",
                    "status": "needs_clarification",
                    "intent": intent_data.get("task", "unknown"),
                    "message": "请补充需要处理的安全任务或提供更明确的目标。",
                }
                with self._lock:
                    current = self.store.get_run(run_id)
                    if current.status not in TERMINAL_STATES and current.status != "waiting_approval":
                        current.output = output
                        current.updated_at = utc_now()
                        self.store.update_run(current)
                        self.transition(run_id, "succeeded")
                        self.emit(run_id, "run.completed", {"output_keys": sorted(output.keys())})
                return
            if engine == "workflow":
                output = self._execute_workflow(run, profile)
            elif engine == "opencode":
                output = self._execute_react(run, profile)
            else:
                output = self._execute_prompt_chain(run, profile)
            with self._lock:
                self.check_runtime_limits(run_id)
                current = self.store.get_run(run_id)
                if current.status in TERMINAL_STATES or current.status == "waiting_approval":
                    return
                current.output = output
                current.updated_at = utc_now()
                self.store.update_run(current)
                self.transition(run_id, "succeeded")
                if self.store.get_run(run_id).status == "succeeded":
                    self.emit(run_id, "run.completed", {"output_keys": sorted(output.keys())})
        except ApprovalRequired:
            # request_approval published the durable notice and paused state
            # together before unwinding the executor. Do not notify twice.
            pass
        except RunTimeout as exc:
            self._timeout_run(run_id, exc.scope)
        except RunCancelled as exc:
            self._fail(run_id, str(exc), code="CANCELLED", category="execution", retryable=False)
        except PolicyDenied as exc:
            self._fail(run_id, str(exc), code="POLICY_DENIED", category="policy", retryable=False)
        except ToolNotFound as exc:
            self._fail(run_id, str(exc), code="NOT_FOUND", category="tool", retryable=False)
        except ModelGatewayError as exc:
            self._fail(run_id, str(exc), code=exc.code, category=exc.category, retryable=exc.retryable)
        except EventSourceError as exc:
            self._fail(run_id, str(exc), code=exc.code, category=exc.category, retryable=exc.retryable)
        except KnowledgeEmbeddingError as exc:
            self._fail(run_id, str(exc), code="KNOWLEDGE_EMBEDDING_ERROR", category="knowledge", retryable=True)
        except OpenCodeResourceError as exc:
            self._fail(
                run_id,
                str(exc),
                code=getattr(exc, "code", "OPENCODE_RESOURCE_ERROR"),
                category="opencode",
                retryable=False,
            )
        except (KnowledgeAccessDenied, KnowledgeNotFoundError, KnowledgeValidationError) as exc:
            self._fail(run_id, str(exc), code="KNOWLEDGE_ERROR", category="knowledge", retryable=False)
        except (NotFoundError, ValueError, RuntimeErrorBase) as exc:
            self._fail(run_id, str(exc), code="VALIDATION_ERROR", category="validation", retryable=False)
        except Exception as exc:  # pragma: no cover - defensive boundary
            self._fail(run_id, f"unexpected runtime error: {exc}", code="INTERNAL", category="system", retryable=False)

    def _emit_decision_events(self, run_id: str, snapshot: Dict[str, Any]) -> None:
        if not isinstance(snapshot, dict) or not snapshot.get("decision_id") or snapshot.get("_events_emitted"):
            return
        intent = snapshot.get("intent") if isinstance(snapshot.get("intent"), dict) else {}
        plan = snapshot.get("plan") if isinstance(snapshot.get("plan"), dict) else {}
        validation = snapshot.get("validation") if isinstance(snapshot.get("validation"), dict) else {}
        route = snapshot.get("route") if isinstance(snapshot.get("route"), dict) else {}
        decision_id = snapshot.get("decision_id")
        plan_id = plan.get("plan_id")
        self.emit(
            run_id,
            "decision.intent.detected",
            {
                "decision_id": decision_id,
                "intent": intent.get("task"),
                "confidence": intent.get("confidence"),
                "source": intent.get("source"),
            },
        )
        self.emit(
            run_id,
            "decision.plan.created",
            {
                "decision_id": decision_id,
                "plan_id": plan_id,
                "plan_hash": snapshot.get("plan_hash"),
                "step_count": len(plan.get("steps") or []) if isinstance(plan.get("steps"), list) else 0,
                "strategy": plan.get("strategy"),
            },
        )
        self.emit(
            run_id,
            "decision.plan.validated",
            {
                "plan_id": plan_id,
                "valid": validation.get("valid", True),
                "error_count": len(validation.get("errors") or []),
                "warning_count": len(validation.get("warnings") or []),
                "approval_required": validation.get("approval_required") or [],
            },
        )
        self.emit(
            run_id,
            "decision.route.selected",
            {
                "decision_id": decision_id,
                "engine": route.get("engine"),
                "source": route.get("source"),
                "reason": route.get("reason"),
            },
        )
        with self._lock:
            try:
                current = self.store.get_run(run_id)
                persisted = dict(current.decision_snapshot or {})
                persisted["_events_emitted"] = True
                current.decision_snapshot = persisted
                current.updated_at = utc_now()
                self.store.update_run(current)
            except NotFoundError:
                return

    def cancel(self, run_id: str) -> Run:
        return self.cancel_for_tenant(run_id)

    def cancel_for_tenant(
        self,
        run_id: str,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        roles: Optional[Iterable[str]] = None,
    ) -> Run:
        with self._lock:
            run = self.store.get_run(run_id)
            self._assert_run_access(run, tenant_id=tenant_id, user_id=user_id, roles=roles)
            if run.status not in TERMINAL_STATES:
                self._cancel_events.setdefault(run_id, threading.Event()).set()
                self._set_run_error(run_id, "CANCELLED", "execution", False, "run cancelled", None)
                self._rollback_checkpoint(run_id)
                self._emit_error(run_id, "CANCELLED", "execution", False, "run cancelled")
                self.transition(run_id, "cancelled")
                self.emit(run_id, "run.cancelled", {"reason": "operator_requested"})
            return self.store.get_run(run_id)

    def decide_approval(
        self,
        approval_id: str,
        approved: bool,
        decided_by: str,
        comment: Optional[str],
        *,
        tenant_id: Optional[str] = None,
    ) -> Approval:
        with self._lock:
            approval = self.store.get_approval(approval_id)
            self._assert_tenant(approval.tenant_id, tenant_id)
            run = self.store.get_run(approval.run_id)
            if approval.status != "pending":
                raise ValueError("approval has already been decided")
            if run.status != "waiting_approval":
                raise ValueError("run is no longer awaiting approval")
            self.check_runtime_limits(run.id)
            approval.status = "approved" if approved else "rejected"
            approval.decided_by = decided_by
            approval.comment = comment
            approval.decided_at = utc_now()
            approval = self.store.update_approval(approval)
            self.emit(run.id, "approval.decided", {"approval_id": approval.id, "approved": approved, "decided_by": decided_by})
            if not approved:
                self._fail(run.id, comment or "operator rejected the tool call", code="POLICY_DENIED", category="policy", retryable=False)
                return approval
            if not approval.tool_name.startswith("component:"):
                run.input["_approved_approval_id"] = approval.id
            self.store.update_run(run)
            self.transition(run.id, "running")
            self._futures[run.id] = self._executor.submit(self.execute, run.id)
            return approval

    def list_tools(self) -> List[ToolContract]:
        return self.tools.list_contracts()

    def list_skills(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "sha256": item.sha256,
                "resources": list(item.resources),
            }
            for item in self.skills.list()
        ]

    def get_skill(self, skill_id: str) -> Dict[str, Any]:
        item = self.skills.get(skill_id)
        return {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "sha256": item.sha256,
            "content": item.content,
            "resources": list(item.resources),
        }

    def list_mcp_servers(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": item.id,
                "transport": item.transport,
                "url": item.url,
                "command": item.command,
                "tools": [tool.qualified_name for tool in item.tools],
            }
            for item in self.mcp.list_servers()
        ]

    def list_mcp_tools(self, server_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        return [
            {
                "name": item.qualified_name,
                "server_id": item.server_id,
                "description": item.description,
                "input_schema": item.input_schema,
                "permission": item.permission,
                "risk_level": item.risk_level,
                "requires_approval": item.requires_approval,
            }
            for item in self.mcp.list_tools(server_ids)
        ]

    def list_subagents(self) -> List[Dict[str, Any]]:
        return [
            {"id": item.id, "system_prompt": item.system_prompt, "skills": list(item.skills), "mcp_servers": list(item.mcp_servers), "max_steps": item.max_steps}
            for item in self.subagents.list()
        ]

    def run_subagent_tool(self, run_id: str, subagent_id: str, task: str) -> Dict[str, Any]:
        run = self.store.get_run(run_id)
        profile = self.store.get_profile(run.agent_id, run.agent_version)
        definition = self.subagents.get(subagent_id)
        self.check_runtime_limits(run_id, scope="subagent")
        skill_context = self.skills.prompt_context(definition.skills) if definition.skills else ""
        prompt = definition.system_prompt
        if skill_context:
            prompt += "\n\nSkill reference:\n" + skill_context
        response = self.model_chat_with_retry(
            run_id,
            max_retry=profile.max_retry,
            model=profile.model.chat,
            system_prompt=prompt,
            messages=[{"role": "user", "content": str(task)[:20_000]}],
            temperature=profile.model.temperature,
            max_tokens=profile.model.max_tokens,
        )
        return {"subagent_id": definition.id, "content": response.get("content", ""), "reason_code": response.get("reason_code"), "steps": 1}

    def _register_builtin_subagents(self) -> None:
        if not self.subagents.list():
            self.subagents.register(SubAgentDefinition(id="evidence-collector", system_prompt="只收集和整理证据，不执行副作用工具。", skills=["event-investigation"], max_steps=6))

    def _register_mcp_tools(self) -> None:
        """Expose configured MCP tools through the normal policy boundary."""

        for item in self.mcp.list_tools():
            if any(existing.name == item.qualified_name for existing in self.tools.list_contracts()):
                continue
            self.tools.register(
                mcp_contract(item),
                lambda args, context, name=item.qualified_name: self.mcp.invoke(name, args, context),
            )

    def _register_opencode_tools(self) -> None:
        """Register the constrained file/shell tools used by OpenCode.

        They are intentionally ordinary ToolRegistry entries: profiles choose
        whether to allow, ask, or deny them and the existing approval path is
        therefore mandatory for shell and mutation operations.
        """
        definitions = [
            ("fs.read", "Read a UTF-8 file inside the run sandbox.", "read", {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}, lambda a, c: self.sandbox.read(c.run_id, c.tenant_id, a["path"])),
            ("fs.list", "List a directory inside the run sandbox.", "read", {"type": "object", "properties": {"path": {"type": "string"}}}, lambda a, c: self.sandbox.list(c.run_id, c.tenant_id, a.get("path", "."))),
            ("fs.search", "Search text inside files in the run sandbox.", "read", {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}, "path": {"type": "string"}}}, lambda a, c: self.sandbox.search(c.run_id, c.tenant_id, a["query"], a.get("path", "."))),
            ("fs.write", "Write a UTF-8 file inside the run sandbox.", "edit", {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}, lambda a, c: self.sandbox.write(c.run_id, c.tenant_id, a["path"], a["content"])),
            ("shell.exec", "Execute a bounded shell command in the run sandbox.", "bash", {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}, "timeout_seconds": {"type": "number"}}}, lambda a, c: self.sandbox.exec(c.run_id, c.tenant_id, a["command"], timeout_seconds=float(a.get("timeout_seconds", getattr(self.settings, "opencode_shell_timeout_seconds", 10.0))))),
            ("subagent.run", "Run an allow-listed bounded subagent for an isolated subtask.", "analyze", {"type": "object", "required": ["subagent_id", "task"], "properties": {"subagent_id": {"type": "string"}, "task": {"type": "string"}}}, lambda a, c: self.run_subagent_tool(c.run_id, a["subagent_id"], a["task"])),
        ]
        for name, description, permission, schema, handler in definitions:
            if any(existing.name == name for existing in self.tools.list_contracts()):
                continue
            self.tools.register(
                ToolContract(
                    name=name,
                    description=description,
                    permission=permission,
                    risk_level="high" if permission in {"bash", "edit"} else "low",
                    input_schema=schema,
                    reversible=permission != "bash",
                    requires_approval=permission in {"bash", "edit"},
                ),
                handler,
            )

    def knowledge_context(self, run: Run) -> KnowledgeAccessContext:
        """Build a knowledge ACL context exclusively from the persisted Run."""

        return KnowledgeAccessContext(
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            roles=list(run.roles),
        )

    def retrieve_knowledge(self, run: Run) -> Optional[Dict[str, Any]]:
        """Retrieve bounded, ACL-filtered evidence for a workflow investigation.

        A missing or blank query is treated as no retrieval request.  The
        returned object is deliberately kept as data (including ``untrusted``
        markers) so model prompts cannot confuse knowledge text with policy.
        """

        # Retrieval is explicit.  Ordinary workflow messages must not cause a
        # hidden knowledge lookup, which would both add latency and broaden the
        # evidence surface unexpectedly.  ``use_knowledge=true`` opts in and
        # uses the message/alert as the query when no dedicated query exists.
        raw_query = run.input.get("knowledge_query")
        use_knowledge = run.input.get("use_knowledge") is True
        if raw_query is None and not use_knowledge:
            return None
        if raw_query is None:
            raw_query = run.input.get("message") or run.input.get("alert") or ""
        query = str(raw_query).strip()
        if not query:
            return None
        options = run.input.get("knowledge_options")
        if not isinstance(options, dict):
            options = {}
        top_k = options.get("top_k", 5)
        min_score = options.get("min_score", 0.0)
        include_flagged = bool(options.get("include_flagged", False))
        try:
            result = self.rag.search(
                query,
                context=self.knowledge_context(run),
                top_k=top_k,
                min_score=min_score,
                include_flagged=include_flagged,
                embedding_context=ModelCallContext(
                    timeout_seconds=self._remaining_timeout(run.id),
                    deadline_monotonic=self._deadline_monotonic.get(run.id),
                    cancel_event=self._cancel_events.setdefault(run.id, threading.Event()),
                ),
            )
        except KnowledgeEmbeddingError:
            # If cancellation/deadline won while the provider was returning,
            # preserve the Run's terminal execution state instead of mapping
            # the race to a generic knowledge-service failure.
            self.check_runtime_limits(run.id, scope="knowledge")
            raise
        payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        self.emit(
            run.id,
            "knowledge.search",
            {
                "query_length": len(query),
                "result_count": len(result.results),
                "total_candidates": result.total_candidates,
                "filtered_count": result.filtered_count,
                "warnings": list(result.warnings),
                "abstain": result.abstain,
                "citation_ids": [citation.citation_id for citation in result.citations],
            },
        )
        return payload

    def get_run(
        self,
        run_id: str,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        roles: Optional[Iterable[str]] = None,
    ) -> Run:
        with self._lock:
            run = self.store.get_run(run_id)
            self._assert_run_access(run, tenant_id=tenant_id, user_id=user_id, roles=roles)
            self._ensure_runtime_tracking(run)
            return run

    def event_snapshot(
        self, run_id: str, after_seq: int = 0, *,
        tenant_id: Optional[str] = None, user_id: Optional[str] = None,
        roles: Optional[Iterable[str]] = None,
    ):
        """Return status and events from one publication boundary for SSE.

        Read status first: a later status must never close a stream whose
        earlier event batch did not yet include that status's final notice.
        """
        with self._lock:
            run = self.get_run(run_id, tenant_id=tenant_id, user_id=user_id, roles=roles)
            return run, self.store.list_events(run_id, after_seq)

    def list_events(
        self,
        run_id: str,
        after_seq: int = 0,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        roles: Optional[Iterable[str]] = None,
    ):
        _, events = self.event_snapshot(
            run_id, after_seq, tenant_id=tenant_id, user_id=user_id, roles=roles,
        )
        return events

    def get_approval(self, approval_id: str, *, tenant_id: Optional[str] = None) -> Approval:
        approval = self.store.get_approval(approval_id)
        self._assert_tenant(approval.tenant_id, tenant_id)
        return approval

    def list_dead_letters(self, *, tenant_id: Optional[str] = None) -> List[DeadLetter]:
        items = self.store.list_dead_letters()
        return [item for item in items if tenant_id is None or self.store.get_run(item.run_id).tenant_id == tenant_id]

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        return self.store.list_checkpoints(run_id)

    def list_compensations(self, run_id: Optional[str] = None, *, tenant_id: Optional[str] = None) -> List[Compensation]:
        items = self.store.list_compensations(run_id)
        return [item for item in items if tenant_id is None or self.store.get_run(item.run_id).tenant_id == tenant_id]

    def resolve_compensation(
        self, compensation_id: str, *, action: str, note: Optional[str],
        decided_by: str, tenant_id: Optional[str] = None,
    ) -> Compensation:
        with self._lock:
            items = [item for item in self.store.list_compensations() if item.id == compensation_id]
            if not items:
                raise NotFoundError(f"compensation not found: {compensation_id}")
            item = items[0]
            # Authorize through the owning Run before mutating state or audit.
            self._assert_tenant(self.store.get_run(item.run_id).tenant_id, tenant_id)
            item.status = "resolved" if action != "reject" else "rejected"
            item.action = action
            item.note = note
            item.decided_by = decided_by
            item.resolved_at = utc_now()
            self.emit(item.run_id, "compensation.resolved", {"compensation_id": item.id, "action": action})
            return self.store.update_compensation(item)

    @staticmethod
    def _assert_tenant(actual_tenant: str, requested_tenant: Optional[str]) -> None:
        if requested_tenant is not None and actual_tenant != requested_tenant:
            raise TenantAccessDenied("tenant access is not authorized")

    def _assert_run_access(
        self,
        run: Run,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        roles: Optional[Iterable[str]] = None,
    ) -> None:
        """Enforce tenant scope and same-tenant resource ownership."""

        self._assert_tenant(run.tenant_id, tenant_id)
        if user_id is None:
            return
        role_set = {role for role in (roles or ()) if isinstance(role, str)}
        if role_set.intersection({"admin", "operator"}):
            return
        if run.user_id != user_id:
            raise TenantAccessDenied("run access is not authorized")

    def _execute_workflow(self, run: Run, profile: AgentProfile) -> Dict[str, Any]:
        workflow_id = run.workflow_ref or profile.workflow_ref or profile.id
        workflow = self.workflow_catalog.resolve(workflow_id, run.workflow_version)
        return WorkflowExecutor(self, run, profile, workflow).execute()

    def _register_builtin_components(self) -> None:
        register_builtin_components(
            self.components,
            retrieve_knowledge=self.retrieve_knowledge,
            model_chat=self.model_chat,
        )

    def _execute_react(self, run: Run, profile: AgentProfile) -> Dict[str, Any]:
        return ReactExecutor(self).execute(run, profile)

    def invoke_tool(self, run: Run, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.check_runtime_limits(run.id, scope="tool")
        contract, _ = self.tools.get(tool_name)
        context = ToolInvocationContext(
            run_id=run.id,
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            correlation_id=run.correlation_id,
            roles=list(run.roles),
            deadline_monotonic=self._deadline_monotonic.get(run.id),
            cancel_event=self._cancel_events.setdefault(run.id, threading.Event()),
        )
        self.emit(run.id, "tool.started", {"tool": tool_name})
        # Read/analyze calls can be retried; mutation/edit/bash calls are not
        # retried without an explicit idempotency contract.
        max_retry = int(getattr(contract, "max_retries", 0) or 0)
        retry_policy = str(getattr(contract, "retry_policy", "never") or "never")
        if retry_policy in {"never", "manual"}:
            max_retry = 0
        if contract.permission in {"mutate", "edit", "bash"} or not contract.reversible:
            max_retry = 0
        return self._invoke_with_retry(
            run,
            operation=f"tool:{tool_name}",
            max_retry=max_retry,
            invoke=lambda: self.tools.invoke(tool_name, args, context=context),
            completed_event=("tool.completed", {"tool": tool_name}),
        )

    def invoke_component_with_retry(
        self,
        run: Run,
        profile: AgentProfile,
        context: ComponentContext,
        *,
        max_retry: int,
    ) -> Dict[str, Any]:
        return self._invoke_with_retry(
            run,
            operation=f"node:{context.node_id}",
            max_retry=max_retry,
            invoke=lambda: self.components.invoke(context),
            completed_event=None,
        )

    def _invoke_with_retry(
        self,
        run: Run,
        *,
        operation: str,
        max_retry: int,
        invoke: Callable[[], Dict[str, Any]],
        completed_event: Optional[tuple[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Bounded retry for transient failures within one node/tool."""

        attempts = max(0, min(int(max_retry), 10))
        for attempt in range(attempts + 1):
            self.check_runtime_limits(run.id, scope=operation)
            try:
                result = invoke()
                if not isinstance(result, dict):
                    raise ValueError(f"{operation} must return an object")
                self.check_runtime_limits(run.id, scope=operation)
                if completed_event:
                    event_type, data = completed_event
                    self.emit(run.id, event_type, {**data, "output_keys": sorted(result.keys())})
                return result
            except Exception as exc:
                retryable, retry_after_ms = self.retry_decision(exc, operation)
                if not retryable or attempt >= attempts:
                    raise
                delay = self.retry_delay(retry_after_ms, attempt)
                self.emit(
                    run.id,
                    "node.retrying" if operation.startswith("node:") else "tool.retrying",
                    {"operation": operation, "attempt": attempt + 1, "max_attempts": attempts + 1, "delay_ms": int(delay * 1000), "reason": str(exc)[:256]},
                )
                self.transition(run.id, "retrying")
                self._wait_retry(run.id, delay)
                self.transition(run.id, "running")
        raise RuntimeError(f"{operation} retry loop exhausted")

    @staticmethod
    def retry_decision(exc: Exception, operation: str) -> tuple[bool, Optional[int]]:
        if isinstance(exc, (ModelGatewayError, EventSourceError)):
            return bool(getattr(exc, "retryable", False)), getattr(exc, "retry_after_ms", None)
        if isinstance(exc, KnowledgeEmbeddingError):
            return True, None
        return False, None

    def retry_delay(self, retry_after_ms: Optional[int], attempt: int) -> float:
        if retry_after_ms is not None:
            return max(0.0, min(float(retry_after_ms) / 1000.0, 30.0))
        base = max(0.0, float(getattr(self.settings, "model_gateway_backoff_seconds", 0.25)))
        return min(30.0, base * (2 ** attempt) + random.uniform(0.0, base))

    def _wait_retry(self, run_id: str, delay: float) -> None:
        if delay <= 0:
            self.check_runtime_limits(run_id, scope="retry")
            return
        event = self._cancel_events.setdefault(run_id, threading.Event())
        if event.wait(min(delay, self._remaining_timeout(run_id))):
            raise RunCancelled("run cancelled")
        self.check_runtime_limits(run_id, scope="retry")

    def request_approval(self, approval: Approval) -> NoReturn:
        """Publish an actionable pause after its notification/audit is durable.

        Executors must persist their resume checkpoint before calling this.
        Runtime readers and approval decisions share this single-process lock.
        Writing the notice first also prevents a stored waiting status from
        preceding its audit. This is not a distributed database transaction.
        """
        with self._lock:
            self.check_runtime_limits(approval.run_id)
            run = self.store.get_run(approval.run_id)
            pending = self.store.get_approval(approval.id)
            if (run.status != "running" or pending.status != "pending"
                    or pending.run_id != run.id or pending.tenant_id != run.tenant_id
                    or pending.requested_by != run.user_id):
                raise PolicyDenied("approval does not match a pending action on this run")
            self.emit(run.id, "approval.required", {"approval_id": pending.id, "tool": pending.tool_name})
            # A slow audit write must not allow a pause past the absolute deadline.
            self.check_runtime_limits(run.id)
            self.transition(run.id, "waiting_approval")
        raise ApprovalRequired(pending)

    _raise_approval = request_approval

    def _execute_prompt_chain(self, run: Run, profile: AgentProfile) -> Dict[str, Any]:
        return execute_prompt_chain(self, run, profile)

    def model_chat(
        self,
        run_id: str,
        *,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Invoke a model with the Run's remaining wall-clock budget."""

        self.check_runtime_limits(run_id, scope="model")
        context = ModelCallContext(
            timeout_seconds=self._remaining_timeout(run_id),
            deadline_monotonic=self._deadline_monotonic.get(run_id),
            cancel_event=self._cancel_events.setdefault(run_id, threading.Event()),
        )
        try:
            chat_with_context = getattr(self.models, "chat_with_context", None)
            if callable(chat_with_context):
                response = chat_with_context(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context,
                )
            else:  # Compatibility with pre-context custom adapters.
                response = self.models.chat(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except ModelGatewayError:
            # Prefer the Run's terminal reason when the provider error raced
            # with our watchdog or an operator cancellation.
            self.check_runtime_limits(run_id, scope="model")
            raise
        self.check_runtime_limits(run_id, scope="model")
        return response

    def model_chat_with_retry(self, run_id: str, *, max_retry: int, **kwargs: Any) -> Dict[str, Any]:
        run = self.store.get_run(run_id)
        return self._invoke_with_retry(
            run,
            operation="model",
            max_retry=max_retry,
            invoke=lambda: self.model_chat(run_id, **kwargs),
            completed_event=None,
        )

    @staticmethod
    def decision_prompt(run: Run) -> str:
        """Give model-backed engines the frozen goal and plan context."""

        snapshot = run.decision_snapshot or {}
        return str(
            {
                "message": run.input.get("message", run.input),
                "decision": snapshot,
            }
        )

    def _remaining_timeout(self, run_id: str) -> Optional[float]:
        deadline = self._deadline_monotonic.get(run_id)
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def ensure_tool_matches_plan(self, run: Run, tool_name: str) -> None:
        """Keep an explicit ReAct action inside the server-generated plan."""

        snapshot = run.decision_snapshot or {}
        if not isinstance(snapshot, dict):
            return
        plan = snapshot.get("plan")
        if not isinstance(plan, dict):
            return
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return
        planned_tools = {
            str(step.get("tool_name"))
            for step in steps
            if isinstance(step, dict) and step.get("kind") == "tool" and step.get("tool_name")
        }
        if planned_tools and tool_name not in planned_tools:
            raise PolicyDenied("tool call is outside the decision plan")

        # The plan captures the arguments used during validation. Keep the
        # explicit action bound to that snapshot so a caller cannot swap in a
        # different low-risk operation after planning.
        for step in steps:
            if not isinstance(step, dict) or step.get("kind") != "tool" or step.get("tool_name") != tool_name:
                continue
            planned_args = step.get("args") if isinstance(step.get("args"), dict) else {}
            requested = run.input.get("tool_call")
            requested_args = requested.get("args") if isinstance(requested, dict) and isinstance(requested.get("args"), dict) else {}
            if planned_args != requested_args:
                raise PolicyDenied("tool arguments are outside the decision plan")
            break

    def snapshot_step_id(self, run: Run, preferred: str, *, fallback: str) -> str:
        snapshot = run.decision_snapshot or {}
        if not isinstance(snapshot, dict):
            return fallback
        plan = snapshot.get("plan")
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if not isinstance(steps, list):
            return fallback
        ids = {str(step.get("step_id")) for step in steps if isinstance(step, dict) and step.get("step_id")}
        return preferred if preferred in ids else fallback

    def step_started(self, run_id: str, step_id: str, component: str) -> None:
        self._set_step_state(run_id, step_id, "running", component=component)

    def step_waiting(self, run_id: str, step_id: str, component: str) -> None:
        self._set_step_state(run_id, step_id, "waiting_approval", component=component)

    def step_completed(
        self,
        run_id: str,
        step_id: str,
        component: str,
        *,
        output_keys: Optional[List[str]] = None,
    ) -> None:
        self._set_step_state(run_id, step_id, "succeeded", component=component, output_keys=output_keys or [])

    def step_failed(self, run_id: str, step_id: str, component: str) -> None:
        self._set_step_state(run_id, step_id, "failed", component=component)

    def _set_step_state(
        self,
        run_id: str,
        step_id: str,
        status: str,
        *,
        component: str,
        output_keys: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            try:
                run = self.store.get_run(run_id)
            except NotFoundError:
                return
            state = dict(run.execution_state or {})
            steps = dict(state.get("steps") or {})
            record = dict(steps.get(step_id) or {})
            record.update({"status": status, "component": component})
            if output_keys:
                record["output_keys"] = list(output_keys)
            steps[step_id] = record
            state["steps"] = steps
            state["current_step"] = step_id
            state["status"] = status
            if status == "succeeded":
                completed = list(state.get("completed_steps") or [])
                if step_id not in completed:
                    completed.append(step_id)
                state["completed_steps"] = completed
                state["last_successful_step"] = step_id
            elif status == "failed":
                state["failed_step"] = step_id
            run.execution_state = state
            run.updated_at = utc_now()
            self.store.update_run(run)
        event_suffix = {
            "running": "started",
            "succeeded": "completed",
            "failed": "failed",
            "waiting_approval": "waiting_approval",
        }.get(status, status)
        self.emit(
            run_id,
            f"plan.step.{event_suffix}",
            {"step_id": step_id, "component": component, "output_keys": output_keys or []},
        )

    def _rollback_checkpoint(self, run_id: str) -> None:
        """Record a best-effort in-memory rollback point on failure."""

        with self._lock:
            try:
                run = self.store.get_run(run_id)
            except NotFoundError:
                return
            state = dict(run.execution_state or {})
            current_step = state.get("current_step")
            last_successful = state.get("last_successful_step")
            if not current_step and not last_successful:
                return
            state["status"] = "rolled_back"
            state["rollback_to"] = last_successful
            run.execution_state = state
            run.updated_at = utc_now()
            self.store.update_run(run)
        self.emit(
            run_id,
            "plan.rollback",
            {"failed_step": current_step, "rollback_to": last_successful, "mode": "memory_checkpoint"},
        )

    def transition(self, run_id: str, target: str) -> Run:
        with self._lock:
            run = self.store.get_run(run_id)
            if run.status == target:
                return run
            if run.status in TERMINAL_STATES:
                return run
            run.status = transition(run.status, target)
            run.updated_at = utc_now()
            self.store.update_run(run)
            self.emit(run_id, "run.status", {"status": target})
            if target in TERMINAL_STATES:
                self._clear_timeout(run_id)
            return run

    def _fail(
        self,
        run_id: str,
        message: str,
        *,
        code: str = "INTERNAL",
        category: str = "system",
        retryable: bool = False,
    ) -> None:
        try:
            with self._lock:
                run = self.store.get_run(run_id)
                if run.status not in TERMINAL_STATES:
                    self._rollback_checkpoint(run_id)
                    self._set_run_error(run_id, code, category, retryable, message, None)
                    self._emit_error(run_id, code, category, retryable, message)
                    if code not in {"CANCELLED", "TIMEOUT", "POLICY_DENIED", "VALIDATION_ERROR", "NOT_FOUND"}:
                        self.store.enqueue_dead_letter(
                            DeadLetter(
                                id=uuid.uuid4().hex,
                                run_id=run_id,
                                reason=message,
                                error_code=code,
                                error_category=category,
                                attempts=1,
                                payload={"status": "failed"},
                            )
                        )
                    try:
                        requested = run.input.get("tool_call") if isinstance(run.input, dict) else None
                        tool_name = requested.get("tool") if isinstance(requested, dict) else None
                        if tool_name:
                            contract, _ = self.tools.get(str(tool_name))
                            if contract.permission in {"mutate", "edit", "bash"}:
                                self.store.create_compensation(
                                    Compensation(id=uuid.uuid4().hex, run_id=run_id, action="inspect")
                                )
                    except Exception:
                        pass
                    self.transition(run_id, "failed")
                    self.emit(run_id, "run.failed", {"error_code": code, "error": message})
        except (NotFoundError, InvalidTransition):
            return

    def check_runtime_limits(self, run_id: str, *, scope: str = "run") -> None:
        run = self.store.get_run(run_id)
        self._ensure_runtime_tracking(run)
        if run.status == "timed_out":
            raise RunTimeout(run.timeout_scope or scope)
        if run.status == "cancelled":
            raise RunCancelled("run cancelled")
        event = self._cancel_events.setdefault(run_id, threading.Event())
        if event.is_set():
            raise RunCancelled("run cancelled")
        deadline = self._deadline_monotonic.get(run_id)
        if deadline is not None and time.monotonic() >= deadline:
            # This is the absolute Run deadline, regardless of the caller.
            self._timeout_run(run_id, "run")
            raise RunTimeout("run")

    def _recover_persisted_runs(self) -> None:
        """Recreate local cancellation/deadline guards for persisted runs.

        A monotonic deadline cannot survive a process restart, so it is
        reconstructed from the persisted wall-clock ``deadline_at``. This
        does not claim ownership of queued/running work; a production queue
        must add leases before automatic cross-process execution recovery.
        """

        list_runs = getattr(self.store, "list_runs", None)
        if not callable(list_runs):
            return
        for run in list_runs():
            if run.status in {"queued", "running", "retrying"}:
                self._ensure_runtime_tracking(run)
                self.store.enqueue_run(run.id)
                self._futures[run.id] = self._executor.submit(self._process_queued_run, run.id)

    def _ensure_runtime_tracking(self, run: Run) -> None:
        with self._lock:
            self._cancel_events.setdefault(run.id, threading.Event())
            if run.status in TERMINAL_STATES or run.deadline_at is None:
                return
            if run.id in self._deadline_monotonic:
                return
            deadline_at = run.deadline_at
            if deadline_at.tzinfo is None:
                # Interpret legacy naive timestamps as UTC during migration.
                deadline_at = deadline_at.replace(tzinfo=timezone.utc)
            remaining = max(0.0, (deadline_at - utc_now()).total_seconds())
            self._deadline_monotonic[run.id] = time.monotonic() + remaining
            if run.id not in self._timeout_timers:
                timer = threading.Timer(remaining, self._timeout_run, args=(run.id, "run"))
                timer.daemon = True
                self._timeout_timers[run.id] = timer
                timer.start()

    def _timeout_run(self, run_id: str, scope: str = "run") -> None:
        with self._lock:
            try:
                run = self.store.get_run(run_id)
            except NotFoundError:
                return
            if run.status in TERMINAL_STATES:
                self._clear_timeout(run_id)
                return
            # A node/model deadline may coincide with the global watchdog.
            # Global exhaustion always wins, making attribution deterministic.
            deadline = self._deadline_monotonic.get(run_id)
            if deadline is not None and time.monotonic() >= deadline:
                scope = "run"
            self._cancel_events.setdefault(run_id, threading.Event()).set()
            self._set_run_error(run_id, "TIMEOUT", "execution", False, f"run timed out ({scope})", scope)
            self._rollback_checkpoint(run_id)
            self._emit_error(run_id, "TIMEOUT", "execution", False, f"run timed out ({scope})")
            try:
                self.transition(run_id, "timed_out")
            except InvalidTransition:
                return
            self.emit(run_id, "run.timed_out", {"timeout_scope": scope})

    def _set_run_error(
        self,
        run_id: str,
        code: str,
        category: str,
        retryable: bool,
        message: str,
        timeout_scope: Optional[str],
    ) -> None:
        run = self.store.get_run(run_id)
        run.error = redact_text(message, max_length=512)
        run.error_code = code
        run.error_category = category
        run.error_retryable = retryable
        if timeout_scope:
            run.timeout_scope = timeout_scope
        run.updated_at = utc_now()
        self.store.update_run(run)

    def _clear_timeout(self, run_id: str) -> None:
        timer = self._timeout_timers.pop(run_id, None)
        if timer is not None:
            timer.cancel()
        self._deadline_monotonic.pop(run_id, None)

    def emit(self, run_id: str, event_type: str, data: Dict[str, Any]) -> None:
        run = self.store.get_run(run_id)
        payload = redact_value(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "tenant_id": run.tenant_id,
                "correlation_id": run.correlation_id,
                **data,
            }
        )
        event = self.store.append_event(run_id, event_type, payload)
        self.store.append_audit(
            {
                "run_id": run_id,
                "tenant_id": run.tenant_id,
                "correlation_id": run.correlation_id,
                "event_seq": event.seq,
                "event_type": event_type,
                "data": payload,
                "created_at": utc_now().isoformat(),
            }
        )

    def _emit_error(self, run_id: str, code: str, category: str, retryable: bool, message: str) -> None:
        self.emit(
            run_id,
            "run.error",
            {
                "error": {
                    "code": code,
                    "category": category,
                    "message": redact_text(message, max_length=512),
                    "retryable": retryable,
                }
            },
        )

    def save_execution_state(self, run_id: str, namespace: str, state: Dict[str, Any]) -> None:
        """Atomically replace one executor's checkpoint without losing other run state."""
        with self._lock:
            current = self.store.get_run(run_id)
            current.execution_state[namespace] = deepcopy(state)
            self.store.update_run(current)

    def deadline_for(self, run_id: str) -> float:
        """Absolute monotonic deadline; infinity means no run-level deadline."""
        return self._deadline_monotonic.get(run_id, float("inf"))

    # Compatibility for integrations using the former private helper names.
    _check_runtime_limits = check_runtime_limits
    _emit = emit
    _step_started = step_started
    _step_waiting = step_waiting
    _step_completed = step_completed
    _step_failed = step_failed
    _transition = transition
    _retry_decision = retry_decision
    _retry_delay = retry_delay
    _invoke_tool = invoke_tool
    _model_chat = model_chat
    _model_chat_with_retry = model_chat_with_retry
    _run_subagent_tool = run_subagent_tool
    _retrieve_knowledge = retrieve_knowledge
    _invoke_component_with_retry = invoke_component_with_retry
    _decision_prompt = decision_prompt
    _ensure_tool_matches_plan = ensure_tool_matches_plan
    _snapshot_step_id = snapshot_step_id
