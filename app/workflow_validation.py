"""Publication validation: graph topology, node bindings and port contracts."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional

from .models import WorkflowManifest, WorkflowValidation
from .component_registry import ComponentRegistry
from .workflow_schema import (
    check_schema, validate_json_schema, version_key, with_defaults,
    _schema_for_port, _schemas_compatible,
)


def validate_workflow_contracts(workflow: WorkflowManifest, registry: "ComponentRegistry") -> List[str]:
    """Check known component schemas and edge port compatibility before running."""

    errors: List[str] = []
    nodes = {node.id: node for node in workflow.nodes}
    for edge in workflow.edges:
        source = nodes.get(edge.from_ref.node)
        target = nodes.get(edge.to_ref.node)
        if source is None or target is None or not registry.has(source.component, source.version) or not registry.has(target.component, target.version):
            continue
        source_definition = registry.definition(source.component, source.version)
        target_definition = registry.definition(target.component, target.version)
        source_properties = source_definition.output_schema.get("properties") if isinstance(source_definition.output_schema, dict) else None
        if edge.from_ref.port != "output" and not (edge.from_ref.port == "error" and source.on_error == "route") and isinstance(source_properties, dict) and edge.from_ref.port not in source_properties:
            errors.append(f"output port not declared: {source.id}.{edge.from_ref.port}")
            continue
        target_contract = target_definition.input_schema
        target_properties = target_contract.get("properties") if isinstance(target_contract, dict) else None
        if isinstance(target_properties, dict) and edge.to_ref.port not in target_properties and target_contract.get("additionalProperties") is False:
            errors.append(f"input port not declared: {target.id}.{edge.to_ref.port}")
            continue
        source_schema = _schema_for_port(source_definition.output_schema, edge.from_ref.port, output=True)
        if source.id == workflow.entry and source.component == "input" and edge.from_ref.port == "output":
            source_schema = workflow.input_schema
        if edge.from_ref.port == "error" and source.on_error == "route":
            source_schema = {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}
        target_schema = _schema_for_port(target_contract, edge.to_ref.port, output=False)
        if target_schema and not _schemas_compatible(source_schema or {}, target_schema):
            errors.append(
                f"incompatible port contract: {source.id}.{edge.from_ref.port} -> "
                f"{target.id}.{edge.to_ref.port}"
            )
    return errors


def validate_workflow(
    workflow: WorkflowManifest,
    component_registry: Optional[ComponentRegistry] = None,
) -> WorkflowValidation:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        version_key(workflow.version)
        check_schema(workflow.input_schema)
        check_schema(workflow.output_schema)
    except ValueError as exc:
        errors.append(str(exc))
    if len(workflow.nodes) > workflow.max_nodes:
        errors.append("workflow exceeds node budget")
    node_ids = [node.id for node in workflow.nodes]
    node_set = set(node_ids)
    if len(node_ids) != len(node_set):
        errors.append("workflow node ids must be unique")
    if workflow.entry not in node_set:
        errors.append(f"entry node not found: {workflow.entry}")
    if workflow.exit not in node_set:
        errors.append(f"exit node not found: {workflow.exit}")

    if component_registry is not None:
        for node in workflow.nodes:
            if not component_registry.has(node.component, node.version):
                errors.append(f"component is not registered: {node.component}@{node.version}")
                continue
            manifest = component_registry.definition(node.component, node.version).manifest
            if (node.component, node.version) in component_registry.disabled:
                errors.append(f"component is disabled: {node.component}@{node.version}")
            try:
                check_schema(node.input_schema)
                errors.extend(f"{node.id}.config: {error}" for error in validate_json_schema(with_defaults(node.config, manifest.config_schema), manifest.config_schema))
            except ValueError as exc:
                errors.append(f"{node.id}: {exc}")
            for dep, version in manifest.dependencies.items():
                if not component_registry.has(dep, version) or (dep, version) in component_registry.disabled:
                    errors.append(f"{node.id}: unresolved dependency {dep}@{version}")
            security = manifest.security
            if security.permission in {"mutate", "edit", "bash"} and not (security.requires_approval and security.compensation):
                errors.append(f"{node.id}: write component requires approval and compensation plan")
            if security.risk_level in {"high", "critical"} and not security.requires_approval:
                errors.append(f"{node.id}: high risk component requires approval")
            if manifest.kind == "control" and node.component not in {"control.router", "control.fork", "control.join", "control.approval", "control.loop"}:
                errors.append(f"{node.id}: unsupported control component")
            if node.component == "control.loop":
                try:
                    body = component_registry.definition(node.config.get("component", ""), node.config.get("version", "")).manifest
                    if body.kind != "function" or not body.execution.idempotent or body.security.permission != "read" or body.security.requires_approval or body.security.risk_level != "low":
                        errors.append(f"{node.id}: loop body must be an idempotent low-risk read function")
                    errors.extend(validate_json_schema(with_defaults(node.config.get("config", {}), body.config_schema), body.config_schema))
                except ValueError as exc:
                    errors.append(f"{node.id}: {exc}")
            if node.component == "model.structured":
                try:
                    check_schema(node.config.get("output_schema", {}))
                except ValueError as exc:
                    errors.append(f"{node.id}: {exc}")
            incoming = {edge.to_ref.port for edge in workflow.edges if edge.to_ref.node == node.id}
            contract = manifest.input_schema
            for port in contract.get("required", []):
                if port not in incoming and port not in node.bindings and "default" not in contract.get("properties", {}).get(port, {}):
                    errors.append(f"{node.id}.{port}: required input has no binding")
            for port, binding in node.bindings.items():
                if port in incoming:
                    errors.append(f"{node.id}.{port}: both edge and binding provided")
                if port not in contract.get("properties", {}) and contract.get("additionalProperties") is False:
                    errors.append(f"{node.id}.{port}: input port not declared")
                if not isinstance(binding, dict) or len(binding) != 1 or not set(binding) <= {"value", "request"}:
                    errors.append(f"{node.id}.{port}: binding must contain value or request")
                elif "value" in binding:
                    errors.extend(f"{node.id}.{port}: {err}" for err in validate_json_schema(binding["value"], contract.get("properties", {}).get(port, {})))
                elif not isinstance(binding["request"], str) or binding["request"] not in workflow.input_schema.get("properties", {}):
                    errors.append(f"{node.id}.{port}: request field is not declared")
                elif not _schemas_compatible(workflow.input_schema["properties"][binding["request"]], contract.get("properties", {}).get(port, {})):
                    errors.append(f"{node.id}.{port}: request binding contract incompatible")
            if node.on_error == "route" and not any(e.from_ref.node == node.id and e.from_ref.port == "error" for e in workflow.edges):
                errors.append(f"{node.id}: error route needs an error edge")

    indegree: Dict[str, int] = {node_id: 0 for node_id in node_set}
    outgoing: Dict[str, List[str]] = defaultdict(list)
    incoming_ports = set()
    for edge in workflow.edges:
        source = edge.from_ref.node
        target = edge.to_ref.node
        if source not in node_set:
            errors.append(f"edge source node not found: {source}")
        if target not in node_set:
            errors.append(f"edge target node not found: {target}")
        port_key = (target, edge.to_ref.port)
        if port_key in incoming_ports:
            errors.append(f"input port has more than one predecessor: {target}.{edge.to_ref.port}")
        incoming_ports.add(port_key)
        if source in node_set and target in node_set:
            outgoing[source].append(target)
            indegree[target] += 1

    if not errors:
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for target in outgoing[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(node_set):
            errors.append("workflow contains an implicit cycle; use an explicit bounded loop component")

    if not errors and node_set:
        reachable = {workflow.entry}
        queue = deque([workflow.entry])
        while queue:
            current = queue.popleft()
            for target in outgoing[current]:
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)
        if len(reachable) != len(node_set):
            missing = sorted(node_set - reachable)
            errors.append("workflow contains nodes unreachable from entry: " + ", ".join(missing))
        if workflow.exit not in reachable:
            errors.append("workflow exit is unreachable from entry")
        reverse = defaultdict(list)
        for source, targets in outgoing.items():
            for target in targets:
                reverse[target].append(source)
        reaching_exit = {workflow.exit}
        queue = deque([workflow.exit])
        while queue:
            for source in reverse[queue.popleft()]:
                if source not in reaching_exit:
                    reaching_exit.add(source)
                    queue.append(source)
        if reaching_exit != node_set:
            errors.append("workflow contains nodes that cannot reach exit")
        if any(e.to_ref.node == workflow.entry or e.from_ref.node == workflow.exit for e in workflow.edges):
            errors.append("entry cannot have predecessors; exit cannot have successors")

    if component_registry is not None and not errors:
        errors.extend(validate_workflow_contracts(workflow, component_registry))

    if not workflow.edges and len(node_set) > 1:
        warnings.append("workflow has multiple nodes but no edges")
    return WorkflowValidation(valid=not errors, errors=errors, warnings=warnings)


def execution_order(workflow: WorkflowManifest) -> List[str]:
    validation = validate_workflow(workflow)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    indegree = {node.id: 0 for node in workflow.nodes}
    outgoing: Dict[str, List[str]] = defaultdict(list)
    for edge in workflow.edges:
        outgoing[edge.from_ref.node].append(edge.to_ref.node)
        indegree[edge.to_ref.node] += 1
    queue = deque([workflow.entry])
    order: List[str] = []
    while queue:
        current = queue.popleft()
        if current in order:
            continue
        order.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(workflow.nodes):
        raise ValueError("workflow contains nodes unreachable from entry")
    return order
