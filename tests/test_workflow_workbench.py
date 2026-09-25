import json
import threading
import time
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.auth import StaticTokenAuthenticator
from app.bootstrap import load_demo_configuration
from app.config import Settings
from app.main import create_app
from app.models import ComponentExecution, ComponentManifest, ComponentResult, ComponentSecurity, RunCreateRequest, WorkflowManifest
from app.runtime import RuntimeService
from app.store import InMemoryStore, SQLiteStore, _parse_model
from app.workflow import ComponentRegistry, _schemas_compatible, check_schema, validate_workflow, validate_json_schema
from pathlib import Path


def manifest(payload):
    return _parse_model(WorkflowManifest, payload)


def edge(source, target, source_port="output", target_port="input"):
    return {"from": {"node": source, "port": source_port}, "to": {"node": target, "port": target_port}}


def completed(runtime, run):
    for _ in range(300):
        current = runtime.store.get_run(run.id)
        if current.status in {"succeeded", "failed", "timed_out", "cancelled", "waiting_approval"}:
            return current
        time.sleep(.01)
    raise AssertionError("run did not finish")


@pytest.fixture
def runtime():
    service = RuntimeService()
    load_demo_configuration(service)
    yield service
    service.close()


def submit(runtime, workflow, inputs=None):
    runtime.publish_workflow(workflow)
    return runtime.submit(RunCreateRequest(tenant_id="tenant-a", user_id="operator", agent_id="event-investigation", input=inputs or {}), workflow=workflow)


def test_schema_direction_defaults_and_semantics():
    assert not _schemas_compatible({"type": ["string", "null"]}, {"type": "string"})
    assert _schemas_compatible({"type": "integer"}, {"type": "number"})
    assert not _schemas_compatible({"type": "object", "properties": {"id": {"type": "string"}}},
        {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]})
    assert not _schemas_compatible({"type": "array", "items": {"type": "string"}}, {"type": "array", "items": {"type": "number"}})
    assert not _schemas_compatible({"type": "string", "x-semantic-type": "asset"}, {"type": "string", "x-semantic-type": "host"})
    assert not _schemas_compatible({"type": "string", "x-sensitivity": "restricted"}, {"type": "string"})
    assert validate_json_schema("300.1.1.1", {"type": "string", "format": "ipv4"})
    with pytest.raises(ValueError, match="unsupported"):
        check_schema({"$ref": "https://untrusted/schema"})


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_immutable_versions_numeric_order_and_catalog_restart(tmp_path, backend):
    store = InMemoryStore() if backend == "memory" else SQLiteStore(tmp_path / "catalog.db")
    wf = manifest({"id": "versioned", "version": "1.2.0", "entry": "x", "exit": "x", "nodes": [{"id": "x", "component": "input"}]})
    store.register_workflow(wf)
    other = deepcopy(wf)
    other.description = "changed"
    with pytest.raises(ValueError, match="immutable"):
        store.register_workflow(other)
    other.version = "1.10.0"
    store.register_workflow(other)
    assert store.get_workflow(wf.id).version == "1.10.0"
    store.catalog_put("draft", "x", "1.0.0", {"a": 1}, expected_revision=0)
    with pytest.raises(ValueError, match="revision"):
        store.catalog_put("draft", "x", "1.0.0", {"a": 2}, expected_revision=0)
    if backend == "sqlite":
        store.close()
        store = SQLiteStore(tmp_path / "catalog.db")
        assert store.catalog_list("draft")[0]["payload"] == {"a": 1}
        store.close()


def test_exact_component_version_and_asset_abstention(runtime):
    registry = runtime.components
    registry.register("version.test", lambda _: {"version": 1})
    registry.register("version.test", lambda _: {"version": 2}, version="2.0.0")
    wf = manifest({"id": "version-test", "entry": "a", "exit": "b", "nodes": [
        {"id": "a", "component": "version.test", "version": "2.0.0"}, {"id": "b", "component": "output"}], "edges": [edge("a", "b")]})
    run = completed(runtime, submit(runtime, wf))
    assert run.status == "succeeded", run.error
    assert run.output["version"] == 2
    asset = runtime.store.get_workflow("asset-safety")
    run = completed(runtime, submit(runtime, asset, {"ip": "10.0.0.1"}))
    assert run.status == "succeeded", run.error
    assert run.output["workflow_status"] == "abstain"
    assert run.output["assessment"] == "insufficient_evidence"
    run = completed(runtime, submit(runtime, asset, {"ip": "10.0.0.1", "events": [{"ip": "10.0.0.1", "tenant_id": "other"}]}))
    assert run.status == "failed"
    assert "tenant" in run.error


def test_parallel_branches_are_concurrent_and_inputs_are_isolated(runtime):
    rendezvous = threading.Barrier(2)
    def work(context):
        assert context.inputs["input"] == {"number": 1}
        context.inputs["input"]["number"] = 2
        rendezvous.wait(timeout=2)
        return {"ok": True}
    runtime.components.register("parallel.test", work)
    wf = manifest({"id": "parallel", "entry": "i", "exit": "j", "nodes": [
        {"id": "i", "component": "input"}, {"id": "a", "component": "parallel.test"},
        {"id": "b", "component": "parallel.test"}, {"id": "j", "component": "control.join"}],
        "edges": [edge("i", "a"), edge("i", "b"), edge("a", "j", target_port="a"), edge("b", "j", target_port="b")]})
    run = completed(runtime, submit(runtime, wf, {"number": 1}))
    assert run.status == "succeeded", run.error
    assert run.input == {"number": 1}


def test_router_skips_unselected_branch_and_any_join_finishes(runtime):
    called = []
    runtime.components.register("selected.test", lambda context: called.append(context.node_id) or {"ok": True})
    wf = manifest({"id": "routing", "entry": "i", "exit": "j", "nodes": [
        {"id": "i", "component": "input"}, {"id": "r", "component": "control.router", "config": {"field": "match", "equals": True}},
        {"id": "a", "component": "selected.test"}, {"id": "b", "component": "selected.test"},
        {"id": "j", "component": "control.join", "join": "any"}],
        "edges": [edge("i", "r"), edge("r", "a", "match"), edge("r", "b", "otherwise"), edge("a", "j", target_port="a"), edge("b", "j", target_port="b")]})
    run = completed(runtime, submit(runtime, wf, {"match": True}))
    assert run.status == "succeeded", run.error
    assert called == ["a"]
    assert run.execution_state["workflow"]["nodes"]["b"]["status"] == "skipped"


def test_approval_resume_does_not_repeat_completed_nodes(runtime):
    calls = []
    runtime.components.register("count.test", lambda ctx: calls.append(1) or {"ok": True})
    wf = manifest({"id": "approval", "entry": "a", "exit": "o", "nodes": [
        {"id": "a", "component": "count.test"}, {"id": "gate", "component": "control.approval"}, {"id": "o", "component": "output"}],
        "edges": [edge("a", "gate"), edge("gate", "o")]})
    run = completed(runtime, submit(runtime, wf))
    assert run.status == "waiting_approval", run.error
    approval = run.execution_state["workflow"]["approval"]["id"]
    runtime.decide_approval(approval, True, "operator", None, tenant_id="tenant-a")
    run = completed(runtime, run)
    assert run.status == "succeeded", run.error
    assert calls == [1]
    assert runtime.store.get_approval(approval).consumed_at


def test_contract_override_cannot_weaken_and_publish_rejects_missing_binding(runtime):
    schema = {"type": "object", "properties": {"ip": {"type": "string"}}, "required": ["ip"], "additionalProperties": False}
    runtime.components.register("strict.test", lambda _: {}, input_schema=schema)
    wf = manifest({"id": "strict", "entry": "a", "exit": "a", "nodes": [{"id": "a", "component": "strict.test", "input_schema": {}}]})
    assert not runtime.validate_workflow(wf).valid
    with pytest.raises(ValueError, match="required"):
        runtime.components.validate_inputs("strict.test", {}, {"type": "object"})


def test_workbench_api_owner_permissions_drafts_and_run(runtime):
    auth = StaticTokenAuthenticator.from_json(json.dumps([
        {"token": "operator-a-token", "subject": "a", "tenant_id": "tenant-a", "roles": ["operator"]},
        {"token": "operator-b-token", "subject": "b", "tenant_id": "tenant-b", "roles": ["operator"]}]))
    client = TestClient(create_app(runtime, auth))
    a = {"Authorization": "Bearer operator-a-token"}
    b = {"Authorization": "Bearer operator-b-token"}
    wf = {"id": "private-flow", "version": "1.0.0", "tenant_id": "tenant-a", "entry": "i", "exit": "o", "nodes": [
        {"id": "i", "component": "input"}, {"id": "o", "component": "output"}], "edges": [edge("i", "o")]}
    assert client.get("/workbench").status_code == 200
    assert client.get("/v1/components").status_code == 401
    assert client.post("/v1/workflows/publish", headers=b, json=wf).status_code == 403
    assert client.put("/v1/workbench/draft", headers=a, json={"workflow": wf}).status_code == 200
    assert client.put("/v1/workbench/draft", headers=a, json={"workflow": wf}).status_code == 422
    assert client.post("/v1/workflows/publish", headers=a, json=wf).status_code == 200
    visible = client.get("/v1/workbench/workflows", headers=b).json()
    assert all(w["id"] != wf["id"] for w in visible["published"])
    assert client.post("/v1/workbench/execute/private-flow/1.0.0", headers=b,
        json={"tenant_id": "tenant-b", "agent_id": "event-investigation"}).status_code == 403
    response = client.post("/v1/workbench/execute/private-flow/1.0.0", headers=a,
        json={"tenant_id": "tenant-a", "user_id": "spoof", "roles": ["admin"], "agent_id": "event-investigation", "input": {"ok": True}})
    assert response.status_code == 202, response.text
    run = completed(runtime, runtime.store.get_run(response.json()["id"]))
    assert run.status == "succeeded", run.error
    assert run.user_id == "a" and run.roles == ["operator"]
    assert client.get(f"/v1/workbench/runs/{run.id}/trace", headers=b).status_code == 403
