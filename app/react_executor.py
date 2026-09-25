"""Bounded ReAct loop: model proposals, tool authorization and resumable state."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .models import AgentProfile, Run, utc_now
from .react_context import ReactContext
from .runtime_errors import PolicyDenied, RunTimeout


class ReactExecutor:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, run: Run, profile: AgentProfile) -> Dict[str, Any]:
        """Run a bounded ReAct loop with persisted conversation state.

        The model is allowed to propose actions, never to authorize them.  A
        pending high-risk call is persisted before entering
        ``waiting_approval``; after approval the loop resumes at that call and
        continues with the same conversation instead of replaying earlier
        actions.
        """
        self.runtime.check_runtime_limits(run.id, scope="react")
        max_steps = max(1, min(int(profile.max_react_steps), int(getattr(self.runtime.settings, "max_react_steps", profile.max_react_steps))))
        self.runtime.emit(run.id, "react.started", {"max_steps": max_steps, "skills": list(profile.skills), "mcp_servers": list(profile.mcp_servers)})
        context = ReactContext(self.runtime, run, profile)
        state = context.state
        step = int(state["step"])
        pending = state["pending_tool"]
        explicit_consumed = bool(state["explicit_consumed"])
        response: Dict[str, Any] = {}

        while step < max_steps:
            self.runtime.check_runtime_limits(run.id, scope="react")
            action: Optional[Dict[str, Any]] = None
            candidate_names: Optional[set[str]] = None
            if pending is not None:
                action = pending
            elif not explicit_consumed and "tool_call" in run.input:
                requested = run.input.get("tool_call")
                explicit_consumed = True
                if not isinstance(requested, dict):
                    raise PolicyDenied("tool_call must be an object")
                action = requested
            else:
                system_prompt, messages, candidate_tools = context.build_prompt()
                candidate_names = set(candidate_tools)
                hook_payload = self.runtime.hooks.run("react.before_model", {"run_id": run.id, "step": step, "messages": messages[-20:], "candidate_tools": candidate_tools})
                messages = list(hook_payload.get("messages") or messages)
                response = self.runtime.model_chat_with_retry(
                    run.id,
                    max_retry=profile.max_retry,
                    model=profile.model.chat,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=profile.model.temperature,
                    max_tokens=profile.model.max_tokens,
                )
                response = self.runtime.hooks.run("react.after_model", {"run_id": run.id, "step": step, "response": response}).get("response", response)
                self.runtime.check_runtime_limits(run.id, scope="model")
                context.record_model_response(response)
                action = self.extract_action(response)
                if action is None:
                    content = str(response.get("content", ""))
                    context.mark_completed(content)
                    self.save_state(run, context.state_for_save(pending_tool=None, explicit_consumed=explicit_consumed, termination_reason="model_final"))
                    self.runtime.emit(run.id, "react.completed", {"steps": step + 1, "reason_code": response.get("reason_code")})
                    observations = context.state["observations"]
                    return {"engine": "opencode", "steps": max(1, step + 1), "content": content, "reason_code": response.get("reason_code"), "tool_observations": observations,
                            "tool_observation": observations[-1]["result"] if observations else None}

            tool_name = str(action.get("tool") or action.get("name") or "")
            if not tool_name:
                raise PolicyDenied("tool call is missing tool name")
            args = action.get("args", action.get("arguments", {}))
            if not isinstance(args, dict):
                raise PolicyDenied("tool_call.args must be an object")
            if tool_name.startswith("mcp.") and tool_name.split(".", 2)[1] not in set(profile.mcp_servers):
                raise PolicyDenied("MCP server is not enabled for this agent")
            if tool_name == "subagent.run" and str(args.get("subagent_id") or "") not in set(profile.subagents):
                raise PolicyDenied("subagent is not enabled for this agent")
            if candidate_names is not None and tool_name not in candidate_names:
                raise PolicyDenied("tool call is outside the current context")
            self.runtime.ensure_tool_matches_plan(run, tool_name)
            react_step_id = f"react.{step + 1}"
            self.runtime.step_started(run.id, react_step_id, tool_name)
            contract, _ = self.runtime.tools.get(tool_name)
            decision = self.runtime.policy_gate.evaluate(profile, contract, args)
            self.runtime.emit(run.id, "policy.checked", {"decision_id": decision.id, "tool": tool_name, "allowed": decision.allowed, "requires_approval": decision.requires_approval, "reason": decision.reason})
            approved_id = run.input.get("_approved_approval_id")
            if approved_id:
                approval = self.runtime.store.get_approval(str(approved_id))
                if (
                    approval.status != "approved"
                    or approval.consumed_at is not None
                    or approval.run_id != run.id
                    or approval.tenant_id != run.tenant_id
                    or approval.requested_by != run.user_id
                    or approval.tool_name != tool_name
                    or approval.args != args
                ):
                    raise PolicyDenied("approval does not match this tool call")
                if not decision.requires_approval and not decision.allowed:
                    raise PolicyDenied(decision.reason)
                self.runtime.store.consume_approval(approval.id, run_id=run.id, tool_name=tool_name, args=args)
            elif not decision.allowed:
                if decision.requires_approval:
                    approval = self.runtime.store.create_approval(run.id, run.tenant_id, tool_name, args, decision.id, run.user_id, run.correlation_id)
                    pending = {"tool": tool_name, "args": args}
                    context.mark_waiting(tool_name)
                    self.save_state(run, context.state_for_save(pending_tool=pending, explicit_consumed=explicit_consumed))
                    self.runtime.step_waiting(run.id, react_step_id, tool_name)
                    self.runtime.request_approval(approval)
                self.runtime.step_failed(run.id, react_step_id, tool_name)
                raise PolicyDenied(decision.reason)

            result = self.runtime.invoke_tool(run, tool_name, args)
            result = self.runtime.hooks.run("react.after_tool", {"run_id": run.id, "step": step + 1, "tool": tool_name, "args": args, "result": result}).get("result", result)
            if approved_id:
                run.input.pop("_approved_approval_id", None)
                self.runtime.store.update_run(run)
            pending = None
            step += 1
            context.state["step"] = step
            context.state["progress"]["phase"] = "investigating"
            context.state["messages"].append({"role": "assistant", "content": json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)})
            context.record_tool(tool_name, args, result)
            observation = context.state["observations"][-1]
            context.state["messages"].append({"role": "tool", "name": tool_name, "content": json.dumps({"observation_id": observation["id"], "summary": observation["summary"], "evidence_refs": observation["evidence_refs"], "raw_result_ref": observation["raw_result_ref"]}, ensure_ascii=False)})
            self.runtime.step_completed(run.id, react_step_id, tool_name, output_keys=sorted(result.keys()))
            self.save_state(run, context.state_for_save(pending_tool=None, explicit_consumed=explicit_consumed))

        self.save_state(run, context.state_for_save(pending_tool=None, explicit_consumed=explicit_consumed, termination_reason="max_steps"))
        raise RunTimeout("react_steps")

    @staticmethod
    def extract_action(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key in ("tool_call", "action"):
            value = response.get(key)
            if isinstance(value, dict):
                return value
        calls = response.get("tool_calls")
        if isinstance(calls, list) and calls and isinstance(calls[0], dict):
            value = calls[0]
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            if function:
                args = function.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                return {"tool": function.get("name"), "args": args}
            return value
        content = response.get("content")
        if isinstance(content, str):
            candidate = content.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict) and (value.get("tool") or value.get("tool_call") or value.get("action")):
                    nested = value.get("tool_call") or value.get("action") or value
                    return nested if isinstance(nested, dict) else None
        return None

    def save_state(self, run: Run, state: Dict[str, Any]) -> None:
        current = self.runtime.store.get_run(run.id)
        execution_state = dict(current.execution_state or {})
        execution_state["react"] = state
        current.execution_state = execution_state
        current.updated_at = utc_now()
        self.runtime.store.update_run(current)
