from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from .models import (
    AgentProfile,
    DecisionSnapshot,
    PlanStep,
    PlanValidation,
    RouteDecision,
    StructuredIntent,
    TaskPlan,
    WorkflowManifest,
    utc_now,
)
from .policy import PolicyGate
from .tools import ToolNotFound, ToolRegistry


KNOWN_COMPONENTS = {
    "input",
    "intent.detect",
    "alert.noise_reduce",
    "event.investigate",
    "report.generate",
    "output",
    "promptchain",
    "tool.validate",
    "result.summarize",
    "clarify",
}


def model_to_dict(value: Any) -> Dict[str, Any]:
    """Return JSON-safe data for both Pydantic major versions."""

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if hasattr(value, "dict"):
        return json.loads(value.json())
    return dict(value)


class DecisionService:
    """Small, deterministic decision layer shared by all execution engines.

    The service intentionally produces a candidate plan only. Tool execution
    remains in RuntimeService, behind PolicyGate and Approval.
    """

    _route_map = {
        "event_investigation": "workflow",
        "security_qa": "promptchain",
        "tool_action": "opencode",
    }

    def __init__(
        self,
        *,
        tool_registry: Optional[ToolRegistry] = None,
        policy_gate: Optional[PolicyGate] = None,
        model_gateway: Optional[Any] = None,
        model_planning_enabled: bool = False,
        max_steps: int = 12,
        model_planner: Optional[Callable[[StructuredIntent, AgentProfile], Optional[TaskPlan]]] = None,
    ) -> None:
        self.tools = tool_registry
        self.policy_gate = policy_gate or PolicyGate()
        self.model_gateway = model_gateway
        self.model_planning_enabled = bool(model_planning_enabled)
        self.max_steps = max(1, int(max_steps))
        self.model_planner = model_planner

    def decide(self, payload: Dict[str, Any], profile: AgentProfile) -> DecisionSnapshot:
        intent = self.recognize(payload)
        plan = self.decompose(intent, payload, profile)
        validation = self.validate_plan(plan, profile)

        # Open-ended tasks may opt into a model-generated candidate. Any
        # parsing or validation problem falls back to the deterministic plan.
        if intent.task == "unknown" and self.model_planning_enabled:
            candidate = self._model_plan(intent, profile)
            if candidate is not None:
                candidate_validation = self.validate_plan(candidate, profile)
                if candidate_validation.valid:
                    plan, validation = candidate, candidate_validation

        route = self.select_route(intent, profile)
        plan_hash = self._plan_hash(plan)
        return DecisionSnapshot(
            decision_id=uuid.uuid4().hex,
            created_at=utc_now(),
            intent=intent,
            plan=plan,
            validation=validation,
            route=route,
            plan_hash=plan_hash,
        )

    def recognize(self, payload: Dict[str, Any]) -> StructuredIntent:
        raw_message = payload.get("message") or payload.get("alert") or payload.get("query") or payload.get("knowledge_query") or ""
        if isinstance(raw_message, (dict, list)):
            raw_message = json.dumps(raw_message, ensure_ascii=False)
        message = str(raw_message).strip()

        explicit_tool = payload.get("tool_call")
        if isinstance(explicit_tool, dict) and explicit_tool.get("tool"):
            args = explicit_tool.get("args") if isinstance(explicit_tool.get("args"), dict) else {}
            return StructuredIntent(
                task="tool_action",
                params={
                    "tool": str(explicit_tool.get("tool")),
                    "args": args,
                    "rewritten_query": message,
                },
                constraints={"explicit_tool_call": True},
                expected_output="tool_observation",
                confidence=1.0,
                source="input",
            )

        # Turn an unambiguous, high-impact natural-language request into a
        # structured candidate action.  This is still only a proposal: the
        # runtime must apply PolicyGate and request approval before execution.
        # Keeping the extraction deterministic prevents free-form model text
        # from becoming an implicit side effect.
        ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", message)
        block_words = ("封禁", "阻断", "拦截", "block", "deny")
        if ip_match and any(word in message.lower() for word in block_words):
            ip = ip_match.group(0)
            valid_ip = all(0 <= int(part) <= 255 for part in ip.split("."))
            if valid_ip:
                reason = message
                return StructuredIntent(
                    task="tool_action",
                    params={
                        "tool": "response.block_ip",
                        "args": {"ip": ip, "reason": reason},
                        "rewritten_query": message,
                    },
                    constraints={
                        "explicit_tool_call": False,
                        "requires_confirmation": True,
                        "read_only": False,
                    },
                    expected_output="tool_observation",
                    confidence=0.94,
                    source="rule",
                )

        lowered = message.lower()
        event_words = ("告警", "事件", "入侵", "incident", "alert", "intrusion")
        attack_words = ("攻击", "attack")
        question_words = ("什么", "如何", "为什么", "查询", "查一下", "分析", "what", "how", "why", "query")
        question_like = any(word in lowered for word in question_words)
        if any(word in lowered for word in event_words) or (any(word in lowered for word in attack_words) and not question_like):
            task = "event_investigation"
            confidence = 0.92
            expected = "security_report"
        elif message and question_like:
            task = "security_qa"
            confidence = 0.72
            expected = "answer"
        elif message:
            task = "security_qa"
            confidence = 0.58
            expected = "answer"
        else:
            task = "unknown"
            confidence = 0.2
            expected = "clarification"

        return StructuredIntent(
            task=task,
            params={"rewritten_query": message or "请补充需要处理的安全任务"},
            constraints={"read_only": task != "tool_action"},
            expected_output=expected,
            confidence=confidence,
            source="rule",
            needs_clarification=task == "unknown" or confidence < 0.4,
        )

    def decompose(
        self,
        intent: StructuredIntent,
        payload: Dict[str, Any],
        profile: AgentProfile,
    ) -> TaskPlan:
        task = intent.task
        if task == "event_investigation":
            steps = [
                self._step("input", "规范化用户输入", component="input", criteria="输入可被后续节点读取"),
                self._step("intent", "识别安全任务意图", component="intent.detect", depends=["input"]),
                self._step("noise", "降低告警噪声并确定优先级", component="alert.noise_reduce", depends=["intent"]),
                self._step("investigation", "结合证据完成事件研判", component="event.investigate", depends=["noise"]),
                self._step("report", "生成结构化安全报告", component="report.generate", depends=["investigation"]),
                self._step("output", "输出最终结果", component="output", depends=["report"]),
            ]
        elif task == "security_qa":
            steps = [
                self._step("intent", "结构化问题和约束", component="intent.detect"),
                self._step("answer", "调用提示链生成回答", component="promptchain", kind="model", depends=["intent"]),
                self._step("output", "输出回答", component="output", depends=["answer"]),
            ]
        elif task == "tool_action":
            tool_call = payload.get("tool_call") if isinstance(payload.get("tool_call"), dict) else {}
            tool_name = str(tool_call.get("tool") or intent.params.get("tool") or "")
            args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else intent.params.get("args", {})
            steps = [
                self._step("validate", "校验工具和参数", component="tool.validate", criteria="工具契约和策略检查通过"),
                self._step(
                    "execute",
                    f"执行工具 {tool_name or '未指定工具'}",
                    kind="tool",
                    tool_name=tool_name,
                    args=args if isinstance(args, dict) else {},
                    depends=["validate"],
                    criteria="工具返回结构化观察结果",
                ),
                self._step("summarize", "汇总工具观察结果", component="result.summarize", kind="model", depends=["execute"]),
            ]
        else:
            steps = [
                self._step("clarify", "向用户补充确认任务目标", component="clarify", criteria="获得明确任务后再执行"),
            ]

        plan_limit = profile.max_react_steps if task == "tool_action" else self.max_steps
        return TaskPlan(
            plan_id=uuid.uuid4().hex,
            goal=intent.params.get("rewritten_query") or intent.task,
            strategy="deterministic_template",
            steps=steps,
            max_steps=min(self.max_steps, max(1, plan_limit)),
            source="template",
        )

    def plan_for_workflow(
        self,
        intent: StructuredIntent,
        workflow: WorkflowManifest,
    ) -> TaskPlan:
        """Mirror a published Workflow manifest into an auditable plan."""

        dependencies: Dict[str, List[str]] = {node.id: [] for node in workflow.nodes}
        for edge in workflow.edges:
            dependencies.setdefault(edge.to_ref.node, []).append(edge.from_ref.node)
        steps = [
            self._step(
                node.id,
                f"执行组件 {node.component}",
                component=node.component,
                depends=dependencies.get(node.id, []),
                criteria="组件返回结构化结果",
            )
            for node in workflow.nodes
        ]
        return TaskPlan(
            plan_id=uuid.uuid4().hex,
            goal=intent.params.get("rewritten_query") or intent.task,
            strategy="workflow_manifest",
            steps=steps,
            max_steps=max(1, len(steps)),
            source="workflow",
        )

    def validate_plan(self, plan: TaskPlan, profile: AgentProfile) -> PlanValidation:
        errors: List[str] = []
        warnings: List[str] = []
        approval_required: List[str] = []
        steps = plan.steps
        ids = [step.step_id for step in steps]
        known_ids = set(ids)
        if not steps:
            errors.append("plan must contain at least one step")
        if len(ids) != len(known_ids):
            errors.append("plan step ids must be unique")
        plan_limit = min(self.max_steps, max(1, profile.max_react_steps)) if any(step.kind == "tool" for step in steps) else self.max_steps
        if plan.source == "workflow":
            plan_limit = max(plan_limit, len(steps))
        if len(steps) > plan_limit:
            errors.append("plan exceeds the maximum step limit")

        graph: Dict[str, List[str]] = {step.step_id: [] for step in steps}
        indegree: Dict[str, int] = {step.step_id: 0 for step in steps}
        for step in steps:
            if not step.objective.strip():
                errors.append(f"step {step.step_id!r} has no objective")
            if not step.success_criteria.strip():
                errors.append(f"step {step.step_id!r} has no success criteria")
            for dependency in step.depends_on:
                if dependency not in known_ids:
                    errors.append(f"step {step.step_id!r} depends on unknown step {dependency!r}")
                else:
                    graph[dependency].append(step.step_id)
                    indegree[step.step_id] += 1

            if step.component and step.component not in KNOWN_COMPONENTS:
                if plan.source == "workflow":
                    warnings.append(f"component validation deferred to workflow runtime: {step.component}")
                else:
                    errors.append(f"unknown component: {step.component}")
            if step.kind == "tool":
                if not step.tool_name:
                    errors.append(f"step {step.step_id!r} has no tool name")
                    continue
                if self.tools is None:
                    warnings.append("tool registry is unavailable; tool validation is deferred")
                    continue
                try:
                    contract, _ = self.tools.get(step.tool_name)
                except ToolNotFound:
                    errors.append(f"unknown tool: {step.tool_name}")
                    continue
                decision = self.policy_gate.evaluate(profile, contract, step.args)
                if decision.requires_approval:
                    approval_required.append(step.step_id)
                    warnings.append(f"step {step.step_id!r} requires operator approval")
                elif not decision.allowed:
                    errors.append(f"tool step {step.step_id!r} is not allowed: {decision.reason}")

        # Kahn's algorithm catches circular dependencies without executing anything.
        queue = [step_id for step_id, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop(0)
            visited += 1
            for target in graph[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(steps):
            errors.append("plan contains a dependency cycle")

        return PlanValidation(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            approval_required=approval_required,
        )

    def select_route(self, intent: StructuredIntent, profile: AgentProfile) -> RouteDecision:
        policy = getattr(profile, "route_policy", "explicit")
        if policy not in {"hybrid", "auto"}:
            return RouteDecision(
                engine=profile.execute_engine,
                profile_id=profile.id,
                profile_version=profile.version,
                workflow_ref=profile.workflow_ref if profile.execute_engine == "workflow" else None,
                reason="explicit profile engine",
                source="profile",
            )
        engine = self._route_map.get(intent.task, profile.execute_engine)
        fallback_reason = ""
        if engine == "workflow" and not profile.workflow_ref:
            engine = profile.execute_engine
            fallback_reason = "; workflow unavailable, kept profile engine"
        return RouteDecision(
            engine=engine,
            profile_id=profile.id,
            profile_version=profile.version,
            workflow_ref=profile.workflow_ref if engine == "workflow" else None,
            reason=f"intent {intent.task} routed by {policy} policy{fallback_reason}",
            source="intent",
        )

    @staticmethod
    def _step(
        step_id: str,
        objective: str,
        *,
        component: str = "",
        kind: str = "component",
        tool_name: str = "",
        args: Optional[Dict[str, Any]] = None,
        depends: Optional[Iterable[str]] = None,
        criteria: str = "步骤输出满足预期",
    ) -> PlanStep:
        return PlanStep(
            step_id=step_id,
            objective=objective,
            kind=kind,
            component=component,
            tool_name=tool_name,
            args=dict(args or {}),
            depends_on=list(depends or []),
            success_criteria=criteria,
        )

    def _model_plan(self, intent: StructuredIntent, profile: AgentProfile) -> Optional[TaskPlan]:
        if self.model_planner is not None:
            try:
                return self.model_planner(intent, profile)
            except Exception:
                return None
        if self.model_gateway is None:
            return None
        try:
            response = self.model_gateway.chat(
                model=profile.model.chat,
                system_prompt="只输出 JSON 计划，不执行工具。",
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "goal": intent.params.get("rewritten_query"),
                                "required_fields": ["goal", "steps"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=min(profile.model.max_tokens, 1200),
            )
            content = str(response.get("content") or "")
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return None
            payload = json.loads(match.group(0))
            raw_steps = payload.get("steps")
            if not isinstance(raw_steps, list):
                return None
            steps = []
            for index, raw in enumerate(raw_steps[: self.max_steps]):
                if not isinstance(raw, dict):
                    continue
                steps.append(
                    self._step(
                        str(raw.get("id") or f"model-{index + 1}"),
                        str(raw.get("objective") or "执行计划步骤"),
                        component=str(raw.get("component") or "clarify"),
                        kind=str(raw.get("kind") or "component"),
                        tool_name=str(raw.get("tool_name") or ""),
                        args=raw.get("args") if isinstance(raw.get("args"), dict) else {},
                        depends=raw.get("depends_on") if isinstance(raw.get("depends_on"), list) else [],
                        criteria=str(raw.get("success_criteria") or "步骤输出满足预期"),
                    )
                )
            if not steps:
                return None
            return TaskPlan(
                plan_id=uuid.uuid4().hex,
                goal=str(payload.get("goal") or intent.params.get("rewritten_query") or intent.task),
                strategy="model_generated",
                steps=steps,
                max_steps=min(self.max_steps, max(1, profile.max_react_steps)),
                source="model",
            )
        except Exception:
            return None

    @staticmethod
    def _plan_hash(plan: TaskPlan) -> str:
        payload = model_to_dict(plan)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
