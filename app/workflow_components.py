"""Installed component adapters and bounded control components."""
from copy import deepcopy
from dataclasses import replace
from functools import partial

from .workflow_schema import check_schema, validate_json_schema
import ipaddress
import json

from .evidence_report import evidence_report
from .models import ComponentExecution, ComponentManifest, ComponentResult, ComponentSecurity


def object_schema(properties=None, required=(), **extra):
    return {"type": "object", "properties": properties or {}, "required": list(required), **extra}


def router(context):
    value = context.inputs["input"]
    candidate = value
    for part in context.config["field"].split("."):
        if not isinstance(candidate, dict) or part not in candidate:
            candidate = None
            break
        candidate = candidate[part]
    port = "match" if candidate == context.config["equals"] else "otherwise"
    return ComponentResult(outputs={port: value}, active_ports=[port])


def loop(context, *, registry):
    body = registry.definition(context.config["component"], context.config["version"]).manifest
    if body.kind != "function" or not body.execution.idempotent or body.security.permission != "read" or body.security.requires_approval or body.security.risk_level != "low":
        raise ValueError("loop body must be an idempotent low-risk read function")
    if body.security.roles and not set(body.security.roles).intersection(context.roles):
        raise ValueError("loop body role is not authorized")
    items = context.inputs["items"]
    if len(items) > context.config["max_iterations"]:
        raise ValueError("loop iteration budget exceeded")
    results = []
    status = "succeeded"
    for item in items:
        context.check_cancelled()
        result = registry.execute(replace(context, component=body.id, version=body.version, inputs={"input": item},
            config=deepcopy(context.config.get("config", {}))))
        if result.status == "failed":
            raise ValueError(result.error or "loop body failed")
        if result.status != "succeeded":
            status = result.status
        results.append(result.outputs)
    return ComponentResult(status=status, outputs={"items": results})


def model_json(context, *, model_chat):
    response = model_chat(context.run_id, model=context.profile.model.chat,
        system_prompt=context.config["prompt"] + "\nReturn a JSON object matching: " + json.dumps(context.config["output_schema"]),
        messages=[{"role": "user", "content": json.dumps(context.inputs["input"], ensure_ascii=False)}],
        temperature=context.profile.model.temperature, max_tokens=context.profile.model.max_tokens)
    schema = context.config["output_schema"]
    check_schema(schema)
    value = json.loads(response["content"])
    errors = validate_json_schema(value, schema)
    if not isinstance(value, dict) or errors:
        raise ValueError("model structured output contract failed")
    # Model confidence is not proof of safety. Missing citations abstains.
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("model evidence must be an array")
    return ComponentResult(status="partial" if evidence else "abstain", outputs={"output": value}, evidence=evidence)


def agent(context, *, run_subagent):
    if context.config["agent_id"] not in context.profile.subagents:
        raise ValueError("subagent is not enabled by this profile")
    value = run_subagent(context.run_id, context.config["agent_id"], context.inputs["task"])
    return ComponentResult(status="partial", outputs={"output": value}, warnings=["Agent output requires evidence review"])


def lookup(context):
    target = str(ipaddress.ip_address(context.inputs["ip"]))
    selected = []
    for record in context.inputs.get("records", []):
        if record.get("tenant_id") != context.tenant_id:
            raise ValueError("evidence record must belong to the current tenant")
        if record.get("ip") == target:
            selected.append(deepcopy(record))
    return ComponentResult(status="partial" if selected else "abstain",
        outputs={"records": selected, "source": "request_supplied_evidence", "ip": target},
        evidence=selected, warnings=[] if selected else ["No evidence for this asset"])


def summarize(context):
    groups = context.inputs
    risks = [r for group in groups.values() for r in group.get("records", []) if r.get("severity") in {"high", "critical"}]
    targets = {group["ip"] for group in groups.values()}
    if len(targets) != 1:
        raise ValueError("risk inputs must refer to the same IP")
    # Absence of findings never proves safety; supplied records are evidence
    # for review, not an authenticated real-time asset inventory.
    assessment = "risk_detected" if risks else "insufficient_evidence"
    return ComponentResult(status="partial" if risks else "abstain", outputs={"assessment": assessment,
        "ip": next(iter(targets)), "evidence_count": sum(len(g["records"]) for g in groups.values()),
        "report": "发现风险证据，需复核处置。" if risks else "证据不足，不能判定资产安全。"}, evidence=risks)


def invoke_installed_tool(ctx, *, tool, invoke_tool):
    # Enforce executable-specific allow-lists inside the adapter so
    # registering an alias cannot bypass the original capability gate.
    if tool.startswith("mcp.") and tool.split(".", 2)[1] not in ctx.profile.mcp_servers:
        raise ValueError("MCP server is not enabled by this profile")
    if tool == "subagent.run" and ctx.inputs.get("subagent_id") not in ctx.profile.subagents:
        raise ValueError("subagent is not enabled by this profile")
    return invoke_tool(ctx.run, tool, ctx.inputs)


def forward_input(context):
    return {"output": context.inputs["input"]}


def join_inputs(context):
    return {"output": context.inputs}


def register_workbench_components(registry, tools, *, invoke_tool, model_chat, run_subagent):
    """Register installed handlers with only their required host capabilities."""

    def install(name, handler, inputs=None, outputs=None, config=None, **kwargs):
        if registry.has(name):
            return
        registry.register(name, handler, manifest=ComponentManifest(id=name, display_name=name,
            input_schema=inputs or {}, output_schema=outputs or {}, config_schema=config or {"type": "object", "additionalProperties": False},
            execution=ComponentExecution(idempotent=True), **kwargs))


    input_port = object_schema({"input": {}}, ["input"], additionalProperties=False)
    install("control.fork", forward_input, input_port,
            object_schema({"output": {}}, ["output"]), kind="control")
    install("control.router", router, input_port,
            object_schema({"match": {}, "otherwise": {}}),
            object_schema({"field": {"type": "string", "minLength": 1}, "equals": {}}, ["field", "equals"], additionalProperties=False), kind="control")
    install("control.join", join_inputs, object_schema(),
            object_schema({"output": {"type": "object"}}, ["output"]), kind="control")
    install("control.approval", lambda ctx: {"output": ctx.inputs["input"]}, input_port,
            object_schema({"output": {}}, ["output"]), kind="control",
            security=ComponentSecurity(requires_approval=True))


    install("control.loop", partial(loop, registry=registry), object_schema({"items": {"type": "array", "items": {}}}, ["items"]),
        object_schema({"items": {"type": "array", "items": {}}}, ["items"]),
        object_schema({"component": {"type": "string"}, "version": {"type": "string"},
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 100}, "config": {"type": "object", "default": {}}},
            ["component", "version", "max_iterations"], additionalProperties=False), kind="control")

    # Tools (including configured HTTP-backed event APIs and MCP) keep their
    # installed handlers and permission contracts. No arbitrary URL or code upload.
    for contract in tools.list_contracts():
        name = "tool." + contract.name
        if registry.has(name):
            continue
        security = ComponentSecurity(permission=contract.permission, risk_level=contract.risk_level,
            requires_approval=contract.requires_approval or contract.permission in {"mutate", "edit", "bash"} or contract.risk_level in {"high", "critical"},
            compensation="inspect_external_effect_before_manual_compensation" if contract.permission in {"mutate", "edit", "bash"} else None)
        registry.register(name, partial(invoke_installed_tool, tool=contract.name, invoke_tool=invoke_tool),
            manifest=ComponentManifest(id=name, display_name=contract.name, description=contract.description, kind="tool",
                input_schema=contract.input_schema, output_schema={"type": "object"}, config_schema={"type": "object", "additionalProperties": False},
                security=security, execution=ComponentExecution(timeout_seconds=contract.timeout_seconds,
                    idempotent=contract.permission in {"read", "analyze"}), implementation=contract.name))


    install("model.structured", partial(model_json, model_chat=model_chat), input_port, object_schema({"output": {"type": "object"}}, ["output"]),
        object_schema({"prompt": {"type": "string"}, "output_schema": {"type": "object"}}, ["prompt", "output_schema"], additionalProperties=False),
        kind="model", security=ComponentSecurity(permission="analyze"))


    install("agent.invoke", partial(agent, run_subagent=run_subagent), object_schema({"task": {"type": "string"}}, ["task"]),
        object_schema({"output": {"type": "object"}}, ["output"]),
        object_schema({"agent_id": {"type": "string"}}, ["agent_id"], additionalProperties=False), kind="agent", security=ComponentSecurity(permission="analyze"))

    ip_schema = {"type": "string", "x-semantic-type": "security.ip.v1"}
    records_schema = {"type": "array", "items": {"type": "object"}, "default": []}
    lookup_input = object_schema({"ip": ip_schema, "records": records_schema, "trigger": {}}, ["ip"], additionalProperties=False)
    finding_schema = object_schema({"records": {"type": "array", "items": {"type": "object"}},
        "source": {"type": "string"}, "ip": ip_schema}, ["records", "source", "ip"])


    for name in ("security.asset.lookup", "security.vulnerability.lookup", "security.event.lookup"):
        install(name, lookup, lookup_input, finding_schema)


    install("security.risk.summarize", summarize,
        object_schema({key: finding_schema for key in ("asset", "vulnerabilities", "events")}, ["asset", "vulnerabilities", "events"], additionalProperties=False),
        object_schema({"assessment": {"type": "string"}, "ip": ip_schema, "evidence_count": {"type": "integer"}, "report": {"type": "string"}},
            ["assessment", "ip", "evidence_count", "report"]))

    # New behavior has a new component ID/version; old publications stay valid.
    install("security.evidence.report", evidence_report,
        object_schema({key: finding_schema for key in ("asset", "vulnerabilities", "events")},
            ["asset", "vulnerabilities", "events"], additionalProperties=False),
        object_schema({"assessment": {"type": "string"}, "ip": ip_schema,
            "evidence_count": {"type": "integer"}, "report": {"type": "string"},
            "findings": {"type": "array"}, "evidence": {"type": "array"}},
            ["assessment", "ip", "evidence_count", "report", "findings", "evidence"]),
        version="1.0.0", description="Deterministic evidence-linked report; unverified input, no safety guarantee")
