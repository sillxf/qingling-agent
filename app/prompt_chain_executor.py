"""Sequential prompt-plan execution; run lifecycle belongs to RuntimeService."""
from __future__ import annotations

from typing import Any, Dict
from .models import AgentProfile, Run
from .component_registry import ComponentContext


def execute_prompt_chain(runtime, run: Run, profile: AgentProfile) -> Dict[str, Any]:
    runtime.check_runtime_limits(run.id, scope="promptchain")
    snapshot = run.decision_snapshot if isinstance(run.decision_snapshot, dict) else {}
    plan = snapshot.get("plan") if isinstance(snapshot.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    outputs: Dict[str, Dict[str, Any]] = {}
    response: Dict[str, Any] = {}
    for raw in steps or [{"step_id": "answer", "kind": "model", "component": "promptchain"}]:
        if not isinstance(raw, dict) or raw.get("step_id") == "output":
            continue
        step_id = str(raw.get("step_id") or "step")
        kind = str(raw.get("kind") or "component")
        component = str(raw.get("component") or "")
        runtime.step_started(run.id, step_id, component or kind)
        try:
            if kind == "model" or component in {"promptchain", "result.summarize"}:
                response = runtime.model_chat_with_retry(
                    run.id,
                    max_retry=profile.max_retry,
                    model=profile.model.chat,
                    system_prompt=profile.system_prompt,
                    messages=[{"role": "user", "content": runtime.decision_prompt(run) + "\n上下文:" + str(outputs)}],
                    temperature=profile.model.temperature,
                    max_tokens=profile.model.max_tokens,
                )
                value = {"content": response.get("content", ""), "reason_code": response.get("reason_code")}
            else:
                value = runtime.invoke_component_with_retry(
                    run,
                    profile,
                    ComponentContext(
                        node_id=step_id,
                        component=component,
                        run=run,
                        profile=profile,
                        inputs={"input": run.input, **outputs},
                        outputs=dict(outputs),
                        config=dict(raw.get("config") or {}),
                    ),
                    max_retry=profile.max_retry,
                )
            outputs[step_id] = value
            runtime.step_completed(run.id, step_id, component or kind, output_keys=sorted(value.keys()))
        except Exception:
            runtime.step_failed(run.id, step_id, component or kind)
            raise
    content = response.get("content") if response else ""
    runtime.emit(run.id, "promptchain.completed", {"steps": len(outputs) or 1})
    output_step_id = runtime.snapshot_step_id(run, "output", fallback="")
    if output_step_id:
        runtime.step_started(run.id, output_step_id, "output")
        runtime.step_completed(run.id, output_step_id, "output", output_keys=["content"])
    return {"engine": "promptchain", "content": content, "reason_code": response.get("reason_code") if response else None, "steps": len(outputs) or 1}
