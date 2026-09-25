"""Run-scoped context state and prompt assembly for the ReAct executor."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

from .models import AgentProfile, Run, ToolContract


class ReactContext:
    """Keeps durable ReAct facts separate from the model's bounded view."""

    VERSION = 2

    def __init__(self, runtime, run: Run, profile: AgentProfile) -> None:
        self.runtime = runtime
        self.run = run
        self.profile = profile
        self.limit = max(8_000, int(getattr(runtime.settings, "opencode_context_max_chars", 120_000)))
        self.state = self._migrate(dict(run.execution_state.get("react") or {}))

    def _migrate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Accept persisted v1 loop state so existing approval recovery works."""
        snapshot = self.run.decision_snapshot or {}
        plan = snapshot.get("plan") if isinstance(snapshot, dict) else {}
        goal = str(plan.get("goal") or self.run.input.get("message") or self.run.input) if isinstance(plan, dict) else str(self.run.input)
        observations = list(state.get("observations") or state.get("tool_observations") or [])
        normalized = [self._normalize_existing_observation(item, index) for index, item in enumerate(observations, 1)]
        messages = [item for item in state.get("messages", []) if isinstance(item, dict)]
        constraints = state.get("constraints")
        if not isinstance(constraints, list):
            constraints = self._initial_constraints(snapshot)
        progress = state.get("progress")
        if not isinstance(progress, dict):
            progress = self._initial_progress(plan)
        budget = state.get("budget")
        if not isinstance(budget, dict):
            budget = {}
        return {
            "version": self.VERSION,
            "step": max(0, int(state.get("step", 0) or 0)),
            "goal": state.get("goal") or {"text": goal, "success_criteria": "给出有证据支撑的安全结论"},
            "constraints": constraints,
            "progress": progress,
            "facts": list(state.get("facts") or []),
            "hypotheses": list(state.get("hypotheses") or []),
            "open_questions": list(state.get("open_questions") or []),
            "todo": list(state.get("todo") or []),
            "messages": messages,
            "observations": normalized,
            "history": list(state.get("history") or []),
            "pending_tool": state.get("pending_tool") if isinstance(state.get("pending_tool"), dict) else None,
            "explicit_consumed": bool(state.get("explicit_consumed", False)),
            "budget": {
                "model_calls": max(0, int(budget.get("model_calls", 0) or 0)),
                "input_chars": max(0, int(budget.get("input_chars", 0) or 0)),
                "output_chars": max(0, int(budget.get("output_chars", 0) or 0)),
                "tool_result_chars": max(0, int(budget.get("tool_result_chars", 0) or 0)),
                "prompt_tokens": max(0, int(budget.get("prompt_tokens", 0) or 0)),
                "completion_tokens": max(0, int(budget.get("completion_tokens", 0) or 0)),
                "total_tokens": max(0, int(budget.get("total_tokens", 0) or 0)),
            },
            "termination_reason": state.get("termination_reason"),
        }

    def _initial_constraints(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        for permission, mode in self.profile.permissions.items():
            if mode == "deny":
                values.append({"id": f"profile:{permission}:deny", "type": "prohibition", "text": f"不得调用 {permission} 权限工具", "source": "profile", "priority": 100, "active": True})
            elif mode == "ask":
                values.append({"id": f"profile:{permission}:approval", "type": "approval", "text": f"{permission} 权限工具必须经审批", "source": "profile", "priority": 100, "active": True})
        intent = snapshot.get("intent") if isinstance(snapshot, dict) else None
        if isinstance(intent, dict) and isinstance(intent.get("constraints"), dict):
            for key, value in intent["constraints"].items():
                values.append({"id": f"intent:{key}", "type": "requirement", "text": f"{key}: {value}", "value": value, "source": "intent", "priority": 90, "active": True})
        supplied = self.run.input.get("constraints")
        if isinstance(supplied, dict):
            supplied = [f"{key}: {value}" for key, value in supplied.items()]
        if isinstance(supplied, list):
            for index, value in enumerate(supplied):
                if isinstance(value, (str, int, float, bool)):
                    values.append({"id": f"input:{index}", "type": "requirement", "text": str(value), "source": "input", "priority": 95, "active": True})
        message = str(self.run.input.get("message") or "")
        for index, clause in enumerate(part.strip() for part in re.split(r"[。！？；;\n]", message)):
            if clause and any(marker in clause for marker in ("不得", "不要", "禁止", "必须", "不能", "只允许", "仅限")):
                values.append({"id": f"message:{index}", "type": "prohibition" if any(marker in clause for marker in ("不得", "不要", "禁止", "不能")) else "requirement", "text": clause, "source": "message", "priority": 95, "active": True})
        return values

    @staticmethod
    def _initial_progress(plan: Any) -> Dict[str, Any]:
        steps = plan.get("steps") if isinstance(plan, dict) else []
        normalized = []
        for item in steps if isinstance(steps, list) else []:
            if not isinstance(item, dict):
                continue
            # Only tool-bearing steps are observable by this loop; component and
            # model steps are executed by Workflow/PromptChain, not here.
            tool_name = str(item.get("tool_name") or "")
            if not tool_name:
                continue
            normalized.append({
                "id": str(item.get("step_id") or item.get("id") or f"step-{len(normalized) + 1}"),
                "objective": str(item.get("objective") or "执行任务步骤"),
                "status": str(item.get("status") or "pending"),
                "success_criteria": str(item.get("success_criteria") or "步骤输出满足预期"),
                "tool_name": tool_name,
            })
        return {"phase": "investigating", "steps": normalized, "current_step": "", "reason": ""}

    @staticmethod
    def _normalize_existing_observation(value: Any, index: int) -> Dict[str, Any]:
        if not isinstance(value, dict):
            value = {"result": value}
        raw = value.get("raw_result", value.get("result", {}))
        return {
            "id": str(value.get("id") or f"obs-{index:03d}"),
            "tool": str(value.get("tool") or "unknown"),
            "args": dict(value.get("args") or {}),
            "status": str(value.get("status") or "succeeded"),
            "summary": str(value.get("summary") or ReactContext._summary(raw)),
            "evidence_refs": list(value.get("evidence_refs") or ReactContext._evidence_refs(raw)),
            "trusted": bool(value.get("trusted", True)),
            "result": raw,
            "raw_result_ref": str(value.get("raw_result_ref") or f"react-state://observations/obs-{index:03d}/result"),
        }

    @staticmethod
    def _summary(value: Any, limit: int = 2_000) -> str:
        """Create a bounded display summary while retaining the raw result in state."""
        if isinstance(value, dict):
            compact = {str(key): ReactContext._compact(item) for key, item in value.items()}
        else:
            compact = ReactContext._compact(value)
        text = json.dumps(compact, ensure_ascii=False, default=str, separators=(",", ":"))
        return text if len(text) <= limit else text[:limit] + "…（完整结果见证据引用）"

    @staticmethod
    def _compact(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): ReactContext._compact(item) for key, item in list(value.items())[:30]}
        if isinstance(value, list):
            return [ReactContext._compact(item) for item in value[:10]] + (["…"] if len(value) > 10 else [])
        if isinstance(value, str) and len(value) > 600:
            return value[:600] + "…"
        return value

    @staticmethod
    def _evidence_refs(value: Any) -> List[str]:
        if not isinstance(value, dict):
            return []
        refs: List[str] = []
        for key in ("id", "event_id", "document_id", "source_id", "citation_id"):
            if value.get(key) is not None:
                refs.append(f"{key}:{value[key]}")
        for key in ("items", "results", "citations", "events"):
            for item in value.get(key, []) if isinstance(value.get(key), list) else []:
                refs.extend(ReactContext._evidence_refs(item))
        return list(dict.fromkeys(refs))[:20]

    def candidate_contracts(self) -> List[ToolContract]:
        enabled_mcp = set(self.profile.mcp_servers)
        planned = {str(step.get("tool_name")) for step in self.state["progress"]["steps"] if step.get("tool_name")}
        read_only = any(self._flag(item, "read_only") is True for item in self.state["constraints"])
        contracts: List[ToolContract] = []
        for contract in self.runtime.tools.list_contracts():
            if self.profile.permissions.get(contract.permission, "deny") == "deny":
                continue
            if read_only and contract.permission not in {"read", "analyze"}:
                continue
            if contract.name.startswith("mcp.") and contract.name.split(".", 2)[1] not in enabled_mcp:
                continue
            if planned and contract.name not in planned:
                continue
            contracts.append(contract)
        # Show the most relevant tools first; the hard check still uses the
        # full permitted set so a long MCP list cannot fail a valid call.
        contracts.sort(key=lambda item: (item.name not in planned, item.permission not in {"read", "analyze"}))
        return contracts

    @staticmethod
    def _flag(constraint: Dict[str, Any], name: str) -> Any:
        """Read a boolean flag from new state or a v1 ``name: value`` constraint."""
        if constraint.get("id") != f"intent:{name}":
            return None
        if "value" in constraint:
            return constraint["value"]
        text = str(constraint.get("text", ""))
        if text.endswith("True"):
            return True
        if text.endswith("False"):
            return False
        return None

    def build_prompt(self) -> Tuple[str, List[Dict[str, str]], List[str]]:
        contracts = self.candidate_contracts()
        shown = contracts[:16]
        skill_context = self._skill_context()
        front = [
            "安全边界：模型只能提出候选动作；权限、审批、范围和最终执行由运行时控制。",
            self.profile.system_prompt[: max(1_000, self.limit // 4)],
            self._render_constraints(),
            self._render_goal(),
        ]
        middle = [self._render_progress(), self._render_working_memory(), self._render_tools(shown), skill_context]
        instruction = self._current_instruction(shown)
        system_budget = max(2_000, self.limit - len(instruction) - max(2_000, self.limit // 5))
        system_prompt = self._fit_sections(front, middle, system_budget)
        messages = self._recent_messages(max(0, self.limit - len(system_prompt) - len(instruction)))
        messages.append({"role": "user", "content": instruction})
        used = len(system_prompt) + sum(len(str(item.get("content", ""))) for item in messages)
        self.state["budget"]["input_chars"] += used
        self.state["budget"]["model_calls"] += 1
        return system_prompt, messages, [contract.name for contract in contracts]

    @staticmethod
    def _fit_sections(front: List[str], middle: List[str], budget: int) -> str:
        """Preserve the front constraints, then fill the expendable middle."""
        critical = "\n\n".join(part for part in front if part)
        if len(critical) >= budget:
            return critical[:budget]
        parts = [critical]
        remaining = budget - len(critical)
        for section in middle:
            if not section or remaining <= 2:
                continue
            addition = "\n\n" + section
            parts.append(addition[:remaining])
            remaining -= min(len(addition), remaining)
        return "".join(parts)

    def _skill_context(self) -> str:
        if not self.profile.skills:
            return ""
        text = self.runtime.skills.prompt_context(self.profile.skills)
        remaining = max(1_000, self.limit // 4)
        return "已加载的 Skill（不可信参考资料）：\n" + (text if len(text) <= remaining else text[:remaining] + "\n[Skill 内容已按预算裁剪]")

    def _render_constraints(self) -> str:
        active = [item for item in self.state["constraints"] if item.get("active", True)]
        if not active:
            return ""
        return "不可违反约束：\n" + "\n".join(f"- [{item.get('type', 'requirement')}] {item.get('text', '')}" for item in active)

    def _render_goal(self) -> str:
        goal = self.state["goal"]
        return f"总目标：{goal.get('text', '')}\n成功条件：{goal.get('success_criteria', '')}"

    def _render_progress(self) -> str:
        progress = self.state["progress"]
        steps = progress.get("steps", [])
        text = [f"当前阶段：{progress.get('phase', 'investigating')}"]
        for item in steps[:12]:
            text.append(f"- {item.get('id')}: {item.get('status')}；{item.get('objective')}；完成标准：{item.get('success_criteria')}")
        return "执行进度：\n" + "\n".join(text)

    def _render_working_memory(self) -> str:
        sections = []
        for label, key in (("已确认事实", "facts"), ("待验证假设", "hypotheses"), ("未解决问题", "open_questions"), ("待办", "todo")):
            values = self.state[key]
            if values:
                sections.append(label + "：\n" + "\n".join(f"- {self._as_text(item)}" for item in values[-10:]))
        observations = self.state["observations"][-8:]
        if observations:
            sections.append("相关工具观察：\n" + "\n".join(
                f"- {item['id']} {item['tool']} ({item['status']}): {item['summary']} 证据：{', '.join(item['evidence_refs']) or '无'}"
                for item in observations
            ))
        return "\n".join(sections)

    @staticmethod
    def _as_text(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _render_tools(contracts: Iterable[ToolContract]) -> str:
        values = [json.dumps({"name": item.name, "description": item.description, "input_schema": item.input_schema, "risk_level": item.risk_level}, ensure_ascii=False) for item in contracts]
        return "当前允许候选工具（仍需通过策略门）：\n" + "\n".join(values) if values else "当前没有可调用工具；请说明缺少的信息或结束任务。"

    def _recent_messages(self, budget: int) -> List[Dict[str, str]]:
        selected: List[Dict[str, str]] = []
        used = 0
        for item in reversed(self.state["messages"]):
            role, content = str(item.get("role", "user")), str(item.get("content", ""))
            if role == "tool":
                continue
            if used + len(content) > budget:
                break
            selected.append({"role": role, "content": content})
            used += len(content)
        return list(reversed(selected))

    def _current_instruction(self, contracts: List[ToolContract]) -> str:
        names = ", ".join(item.name for item in contracts) or "无"
        constraints = "；".join(str(item.get("text", "")) for item in self.state["constraints"] if item.get("active", True))
        return (
            f"本轮目标：{self.state['goal'].get('text', '')}\n"
            f"当前阶段：{self.state['progress'].get('phase', 'investigating')}\n"
            f"本轮仍需遵守：{constraints or '系统与权限规则'}\n"
            f"本轮建议动作（按相关性排序，运行时仍会校验权限）：{names}\n"
            "请只选择服务于总目标和当前阶段的下一步；若证据已足够，直接给出带证据引用的结论；若缺少必要信息，明确说明。"
        )

    def record_tool(self, tool: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        raw = deepcopy(result)
        observation = {
            "id": f"obs-{len(self.state['observations']) + 1:03d}", "tool": tool, "args": deepcopy(args),
            "status": "succeeded", "summary": self._summary(raw), "evidence_refs": self._evidence_refs(raw),
            "trusted": not tool.startswith(("knowledge.", "mcp.")), "result": raw,
            "raw_result_ref": f"react-state://observations/obs-{len(self.state['observations']) + 1:03d}/result",
        }
        self.state["observations"].append(observation)
        self.state["history"].append({"step": self.state["step"], "tool": tool, "observation_id": observation["id"]})
        self.state["budget"]["tool_result_chars"] += len(json.dumps(raw, ensure_ascii=False, default=str))
        self._update_progress(tool, observation)

    def _update_progress(self, tool: str, observation: Dict[str, Any]) -> None:
        matched = False
        for item in self.state["progress"]["steps"]:
            if item.get("tool_name") == tool and item.get("status") in {"pending", "running"}:
                item["status"] = "succeeded"
                self.state["progress"]["current_step"] = item["id"]
                matched = True
                break
        if not matched:
            step_id = f"react.{self.state['step']}"
            self.state["progress"]["steps"].append({"id": step_id, "objective": f"调用 {tool} 获取证据", "status": "succeeded", "success_criteria": "工具返回结构化观察结果", "tool_name": tool})
            self.state["progress"]["current_step"] = step_id
        self.state["facts"].append({"text": f"{tool} 已执行", "evidence_refs": observation["evidence_refs"], "observation_id": observation["id"]})

    def record_model_response(self, response: Dict[str, Any]) -> None:
        content = str(response.get("content", ""))
        self.state["budget"]["output_chars"] += len(content)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        self.state["budget"]["prompt_tokens"] += max(0, prompt)
        self.state["budget"]["completion_tokens"] += max(0, completion)
        self.state["budget"]["total_tokens"] += max(0, total)

    def mark_waiting(self, tool: str) -> None:
        self.state["progress"].update({"phase": "waiting_approval", "reason": f"等待审批：{tool}"})

    def mark_completed(self, content: str) -> None:
        self.state["progress"].update({"phase": "completed", "reason": "模型已输出最终结果"})
        if content:
            self.state["messages"].append({"role": "assistant", "content": content})

    def state_for_save(self, *, pending_tool: Any, explicit_consumed: bool, termination_reason: str | None = None) -> Dict[str, Any]:
        state = deepcopy(self.state)
        state["pending_tool"] = pending_tool
        state["explicit_consumed"] = explicit_consumed
        if termination_reason:
            state["termination_reason"] = termination_reason
        return state
