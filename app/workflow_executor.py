"""Bounded DAG scheduler. Only this coordinator commits node state."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
import threading
import time
import uuid

from .models import Checkpoint, ComponentResult, ToolContract
from .observability import redact_text
from .store import _model_payload
from .component_registry import ComponentContext
from .workflow_schema import manifest_digest, validate_json_schema, with_defaults
from .runtime_errors import ApprovalRequired, PolicyDenied, RunCancelled, RunTimeout


class WorkflowExecutor:
    def __init__(self, runtime, run, profile, workflow):
        self.runtime, self.run, self.profile, self.workflow = runtime, run, profile, workflow
        self.registry = runtime.components
        self.nodes = {n.id: n for n in workflow.nodes}
        self.incoming = {n.id: [e for e in workflow.edges if e.to_ref.node == n.id] for n in workflow.nodes}
        self.state = deepcopy(runtime.store.get_run(run.id).execution_state.get("workflow", {}))
        self.records = self.state.setdefault("nodes", {})
        self.state.setdefault("manifest", manifest_digest(workflow))
        if self.state["manifest"] != manifest_digest(workflow):
            raise ValueError("workflow checkpoint version mismatch")

    def persist(self):
        self.runtime.save_execution_state(self.run.id, "workflow", self.state)

    def inputs(self, node):
        manifest = self.registry.definition(node.component, node.version).manifest
        values = with_defaults({}, manifest.input_schema)
        for port, binding in node.bindings.items():
            if "value" in binding:
                values[port] = deepcopy(binding["value"])
            elif binding["request"] in self.run.input:
                values[port] = deepcopy(self.run.input[binding["request"]])
        for edge in self.incoming[node.id]:
            record = self.records[edge.from_ref.node]
            if record["status"] == "skipped" or (record.get("active_ports") is not None and edge.from_ref.port not in record["active_ports"]):
                continue
            outputs = record["outputs"]
            if edge.from_ref.port in outputs:
                values[edge.to_ref.port] = deepcopy(outputs[edge.from_ref.port])
            elif edge.from_ref.port == "output":
                values[edge.to_ref.port] = deepcopy(outputs)
            else:
                raise ValueError(f"missing output: {edge.from_ref.node}.{edge.from_ref.port}")
        self.registry.validate_inputs(node.component, values, node.input_schema, node.version)
        return values

    def active(self, node):
        if node.id == self.workflow.entry:
            return True
        flags = [self.records[e.from_ref.node]["status"] != "skipped" and (self.records[e.from_ref.node].get("active_ports") is None or e.from_ref.port in self.records[e.from_ref.node]["active_ports"]) for e in self.incoming[node.id]]
        return any(flags) if node.join == "any" else all(flags)

    def authorize(self, node, inputs):
        manifest = self.registry.definition(node.component, node.version).manifest
        security = manifest.security
        if security.roles and not set(security.roles).intersection(self.run.roles):
            raise PolicyDenied("component role is not authorized")
        if manifest.kind == "tool" and manifest.implementation and manifest.implementation.startswith("mcp."):
            server = manifest.implementation.split(".")[1]
            if server not in self.profile.mcp_servers:
                raise PolicyDenied("MCP server is not enabled by this profile")
        contract = ToolContract(name=f"component:{node.component}@{node.version}", permission=security.permission,
                                risk_level=security.risk_level, requires_approval=security.requires_approval)
        args = {"node_id": node.id, "inputs": inputs, "config": node.config, "workflow": self.state["manifest"]}
        decision = self.runtime.policy_gate.evaluate(self.profile, contract, args)
        self.runtime.emit(self.run.id, "policy.checked", {"node_id": node.id, "allowed": decision.allowed, "requires_approval": decision.requires_approval})
        if decision.allowed:
            return
        if not decision.requires_approval:
            raise PolicyDenied(decision.reason)
        pending = self.state.get("approval")
        if pending and pending["node_id"] == node.id:
            approval = self.runtime.store.get_approval(pending["id"])
            if approval.status == "approved" and approval.tenant_id == self.run.tenant_id and approval.requested_by == self.run.user_id:
                self.runtime.store.consume_approval(approval.id, run_id=self.run.id, tool_name=contract.name, args=args)
                self.state.pop("approval", None)
                self.persist()
                return
            self.runtime.request_approval(approval)
        approval = self.runtime.store.create_approval(self.run.id, self.run.tenant_id, contract.name, args, decision.id, self.run.user_id, self.run.correlation_id)
        self.state["approval"] = {"id": approval.id, "node_id": node.id}
        self.persist()
        self.runtime.step_waiting(self.run.id, node.id, node.component)
        self.runtime.request_approval(approval)

    def call(self, node, inputs, cancel, deadline):
        manifest = self.registry.definition(node.component, node.version).manifest
        # Legacy built-ins receive detached snapshots; no component sees the
        # scheduler's mutable run or outputs of unrelated nodes.
        context = ComponentContext(node.id, node.component, deepcopy(self.run), deepcopy(self.profile),
                                   deepcopy(inputs), {}, with_defaults(node.config, manifest.config_schema),
                                   node.version, deadline, cancel)
        retries = min(self.profile.max_retry, manifest.execution.max_retries) if manifest.execution.idempotent and manifest.security.permission in {"read", "analyze"} else 0
        started = time.monotonic()
        for attempt in range(retries + 1):
            context.check_cancelled()
            try:
                result = self.registry.execute(context)
                result.metrics.update({"attempts": attempt + 1, "duration_ms": round((time.monotonic() - started) * 1000)})
                return result
            except Exception as exc:
                retryable, retry_after = self.runtime.retry_decision(exc, "node:" + node.id)
                if attempt >= retries or not retryable:
                    raise
                self.runtime.emit(self.run.id, "node.retrying", {"node_id": node.id, "attempt": attempt + 1})
                cancel.wait(min(self.runtime.retry_delay(retry_after, attempt), max(0, deadline - time.monotonic())))

    def finish(self, node, result):
        if isinstance(result, TimeoutError):
            raise RunTimeout("node:" + node.id)
        if isinstance(result, (PolicyDenied, RunCancelled, RunTimeout, ApprovalRequired)):
            raise result
        if isinstance(result, Exception) or result.status == "failed":
            message = redact_text(str(result) if isinstance(result, Exception) else result.error or "component failed")
            if node.on_error == "fail":
                self.records[node.id] = {"status": "failed", "error": message}
                self.persist()
                if isinstance(result, Exception):
                    raise result
                raise ValueError(message)
            result = ComponentResult(outputs={"error": {"message": message}}, active_ports=["error"] if node.on_error == "route" else [], warnings=[message], status="partial")
        payload = _model_payload(result)
        if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()) > self.workflow.max_output_bytes:
            raise ValueError("component output exceeds artifact budget; use a controlled artifact reference")
        self.records[node.id] = {**payload, "status": "skipped" if result.active_ports == [] else result.status}
        if len(json.dumps(self.records, ensure_ascii=False).encode()) > self.workflow.max_output_bytes * 4:
            raise ValueError("workflow checkpoint data budget exceeded")
        self.persist()
        self.runtime.store.save_checkpoint(Checkpoint(id=uuid.uuid4().hex, run_id=self.run.id, node_id=node.id,
            status=self.records[node.id]["status"], output={"component": node.component, "version": node.version, "metrics": result.metrics}))
        self.runtime.step_completed(self.run.id, node.id, node.component, output_keys=sorted(result.outputs))
        self.runtime.emit(self.run.id, "node.completed", {"node_id": node.id, "component": node.component,
            "version": node.version, "status": result.status, "metrics": result.metrics, "output_keys": sorted(result.outputs)})

    def execute(self):
        errors = validate_json_schema(self.run.input, self.workflow.input_schema)
        if errors:
            raise ValueError("workflow input contract: " + "; ".join(errors))
        if self.workflow.tenant_id and self.workflow.tenant_id != self.run.tenant_id:
            raise ValueError("workflow tenant is not authorized")
        completed = {key for key, record in self.records.items() if record["status"] in {"succeeded", "partial", "abstain", "skipped"}}
        for key, record in self.records.items():
            if record["status"] in {"running", "failed"} and not self.registry.definition(self.nodes[key].component, self.nodes[key].version).manifest.execution.idempotent:
                raise ValueError(f"node {key} has an uncertain outcome; inspect and compensate before a new run")
        pool = ThreadPoolExecutor(max_workers=self.workflow.max_parallel)
        running = {}
        try:
            while len(completed) < len(self.nodes):
                self.runtime.check_runtime_limits(self.run.id, scope="workflow")
                candidates = [node for node in self.workflow.nodes if node.id not in completed and not any(v[0].id == node.id for v in running.values()) and all(e.from_ref.node in completed for e in self.incoming[node.id])]
                for node in candidates:
                    if len(running) >= self.workflow.max_parallel:
                        break
                    if not self.active(node):
                        self.records[node.id] = {"status": "skipped", "outputs": {}, "active_ports": []}
                        completed.add(node.id)
                        self.persist()
                        self.runtime.emit(self.run.id, "node.skipped", {"node_id": node.id})
                        continue
                    manifest = self.registry.definition(node.component, node.version).manifest
                    # Approval is a barrier: outstanding work must checkpoint
                    # before the run can enter waiting_approval.
                    approval_possible = manifest.security.requires_approval or manifest.security.risk_level in {"high", "critical"} or self.profile.permissions.get(manifest.security.permission) == "ask"
                    if approval_possible and running:
                        continue
                    values = self.inputs(node)
                    self.authorize(node, values)
                    self.records[node.id] = {"status": "running", "component": node.component, "version": node.version}
                    self.persist()
                    self.runtime.step_started(self.run.id, node.id, node.component)
                    self.runtime.emit(self.run.id, "node.started", {"node_id": node.id, "component": node.component, "version": node.version})
                    cancel = threading.Event()
                    deadline = min(time.monotonic() + manifest.execution.timeout_seconds, self.runtime.deadline_for(self.run.id))
                    future = pool.submit(self.call, node, values, cancel, deadline)
                    running[future] = (node, cancel, deadline)
                    if approval_possible:
                        break
                if not running:
                    if len(completed) == len(self.nodes):
                        break
                    if not candidates:
                        raise ValueError("workflow cannot make progress")
                    continue
                ready, _ = wait(running, timeout=0.05, return_when=FIRST_COMPLETED)
                for future, (node, cancel, deadline) in list(running.items()):
                    if time.monotonic() >= deadline and future not in ready:
                        cancel.set()
                        self.runtime.check_runtime_limits(self.run.id, scope="node:" + node.id)
                        raise RunTimeout("node:" + node.id)
                    if future in ready:
                        running.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = exc
                        self.finish(node, result)
                        completed.add(node.id)
            record = self.records[self.workflow.exit]
            if record["status"] == "skipped":
                raise ValueError("workflow exit was skipped; add an explicit any join for conditional paths")
            result = deepcopy(record["outputs"])
            errors = validate_json_schema(result, self.workflow.output_schema)
            if errors:
                raise ValueError("workflow output contract: " + "; ".join(errors))
            statuses = {r["status"] for r in self.records.values()}
            result["workflow_status"] = "abstain" if "abstain" in statuses else "partial" if "partial" in statuses else "succeeded"
            result.update(engine="workflow", workflow={"id": self.workflow.id, "version": self.workflow.version})
            return result
        finally:
            for future, (_, cancel, _) in running.items():
                cancel.set()
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
