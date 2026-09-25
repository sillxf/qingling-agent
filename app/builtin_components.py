"""Legacy IPO components, retained under their published names and contracts."""
from functools import partial

from .component_registry import ComponentContext
from .models import ComponentManifest, ComponentExecution


def input_value(context):
    return {"output": context.run.input}


def detect_intent(context):
    snapshot = context.run.decision_snapshot or {}
    intent_data = snapshot.get("intent") if isinstance(snapshot, dict) else None
    if isinstance(intent_data, dict):
        params = intent_data.get("params") if isinstance(intent_data.get("params"), dict) else {}
        return {
            "intent": intent_data.get("task", "security_qa"),
            "confidence": intent_data.get("confidence", 0.0),
            "rewritten_query": params.get("rewritten_query", ""),
            "structured_intent": intent_data,
        }
    source = context.inputs.get("input", context.run.input)
    message = str(source.get("message") or source.get("alert") or "") if isinstance(source, dict) else str(source or "")
    intent = "event_investigation" if any(word in message.lower() for word in ["告警", "事件", "攻击", "alert", "incident"]) else "security_qa"
    return {
        "intent": intent,
        "confidence": 0.92 if intent == "event_investigation" else 0.58,
        "rewritten_query": message.strip() or "请分析输入的安全任务",
    }


def reduce_noise(context):
    source = context.inputs.get("input", context.run.input)
    message = str(source.get("message") or source.get("alert") or "") if isinstance(source, dict) else str(source or "")
    high_risk = any(word in message.lower() for word in ["勒索", "ransomware", "高危", "critical"])
    return {
        "noise_reduction": "pending_model_adapter",
        "priority": "high" if high_risk else "medium",
        "algorithm_plan": ["TextCNN", "DBSCAN", "baseline_learning"],
    }


def investigate_event(context, *, retrieve_knowledge, model_chat):
    knowledge = retrieve_knowledge(context.run)
    response = model_chat(
        context.run_id,
        model=context.profile.model.chat,
        system_prompt=context.profile.system_prompt,
        messages=[{"role": "user", "content": str({
            "input": context.run.input,
            "context": [context.inputs],
            "knowledge_evidence": knowledge,
        })}],
        temperature=context.profile.model.temperature,
        max_tokens=context.profile.model.max_tokens,
    )
    result = {
        "finding": response["content"],
        "evidence": ["intent.detect", "alert.noise_reduce"],
        "confidence": 0.5,
        "abstain": True,
    }
    if knowledge is not None:
        result["knowledge"] = knowledge
    return result


def generate_report(context):
    investigation = context.inputs.get("input", {})
    if not isinstance(investigation, dict) or "finding" not in investigation:
        investigation = next((value for value in context.outputs.values() if "finding" in value), {})
    return {
        "title": "青灵智能体安全事件分析报告",
        "report": "# 青灵智能体安全事件分析报告\n\n"
        + f"- 结论：{investigation.get('finding', '暂无结论')}\n"
        + f"- 证据：{', '.join(investigation.get('evidence', []))}\n"
        + "- 当前状态：演示模式，需接入真实安全数据源和模型后复核。",
    }


def output_value(context):
    value = context.inputs.get("input")
    if not isinstance(value, dict):
        raise ValueError("output component input must be an object")
    return dict(value)


def register_builtin_components(registry, *, retrieve_knowledge, model_chat) -> None:
    """Register the built-ins while leaving custom components injectable."""
    handlers = {
        "input": input_value,
        "intent.detect": detect_intent,
        "alert.noise_reduce": reduce_noise,
        "event.investigate": partial(investigate_event, retrieve_knowledge=retrieve_knowledge, model_chat=model_chat),
        "report.generate": generate_report,
        "output": output_value,
    }
    definitions = {
        "input": ({}, {"type": "object", "additionalProperties": True}),
        "intent.detect": (
            {"type": "object", "additionalProperties": True},
            {"type": "object", "required": ["intent", "confidence", "rewritten_query"], "additionalProperties": True},
        ),
        "alert.noise_reduce": (
            {"type": "object", "additionalProperties": True},
            {"type": "object", "required": ["noise_reduction", "priority", "algorithm_plan"], "additionalProperties": True},
        ),
        "event.investigate": (
            {"type": "object", "additionalProperties": True},
            {"type": "object", "required": ["finding", "evidence", "confidence", "abstain"], "additionalProperties": True},
        ),
        "report.generate": (
            {"type": "object", "additionalProperties": True},
            {"type": "object", "required": ["title", "report"], "additionalProperties": True},
        ),
        "output": (
            {"type": "object", "additionalProperties": True},
            {"type": "object", "additionalProperties": True},
        ),
    }
    for component, (input_schema, output_schema) in definitions.items():
        if not registry.has(component):
            registry.register(
                component,
                handlers[component],
                manifest=ComponentManifest(id=component, input_schema=input_schema, output_schema=output_schema,
                    execution=ComponentExecution(idempotent=True, max_retries=3)),
            )
