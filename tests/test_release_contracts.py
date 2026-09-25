"""Regression gates for deadlines, retries, lifecycle and installable resources."""
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import pytest

from app.bootstrap import load_demo_configuration
from app.config import Settings
from app.main import create_app
from app.model_gateway import DeterministicModelGateway, ModelGatewayTimeout
from app.models import ComponentExecution, ComponentManifest, RunCreateRequest, WorkflowManifest, utc_now
from app.runtime import RunTimeout, RuntimeService


@pytest.fixture
def runtime(tmp_path):
    service = RuntimeService(app_settings=Settings(
        store_backend="memory", knowledge_store_backend="memory", config_dir=None,
        opencode_sandbox_root=str(tmp_path / "sandbox"), run_timeout_seconds=5,
        model_gateway_backoff_seconds=0,
    ))
    load_demo_configuration(service)
    yield service
    service.close()


def wait(runtime, run):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        current = runtime.store.get_run(run.id)
        if current.status in {"succeeded", "failed", "timed_out", "cancelled"}:
            return current
        time.sleep(0.01)
    raise AssertionError("run did not finish within test deadline")


@pytest.mark.parametrize("scope", ["workflow", "model", "node:investigation", "knowledge"])
def test_absolute_deadline_is_attributed_to_run_not_polling_component(runtime, scope):
    profile = runtime.store.get_profile("event-investigation")
    run = runtime.store.create_run(RunCreateRequest(tenant_id="tenant-a", agent_id=profile.id), profile,
                                   deadline_at=utc_now() - timedelta(seconds=1))
    with pytest.raises(RunTimeout):
        runtime.check_runtime_limits(run.id, scope=scope)
    current = runtime.store.get_run(run.id)
    assert current.status == "timed_out"
    assert current.timeout_scope == "run"
    assert current.output is None


def test_node_timeout_keeps_node_scope_when_run_budget_remains(runtime):
    def slow(context):
        context.cancel_event.wait(0.5)
        context.check_cancelled()
        return {"output": {}}

    runtime.components.register("slow.read", slow, manifest=ComponentManifest(
        id="slow.read", input_schema={"type": "object"}, output_schema={"type": "object"},
        execution=ComponentExecution(timeout_seconds=0.02, idempotent=True),
    ))
    workflow = WorkflowManifest(id="node-timeout", version="1.0.0", entry="slow", exit="slow",
                                nodes=[{"id": "slow", "component": "slow.read"}], edges=[])
    runtime.publish_workflow(workflow)
    run = runtime.submit(RunCreateRequest(tenant_id="tenant-a", agent_id="event-investigation"), workflow=workflow)
    final = wait(runtime, run)
    assert final.status == "timed_out"
    assert final.timeout_scope == "node:slow"


def test_model_retry_budget_is_bounded_and_zero_backoff_is_honored(runtime):
    class TimeoutGateway(DeterministicModelGateway):
        calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            raise ModelGatewayTimeout("synthetic upstream timeout")

    gateway = TimeoutGateway()
    runtime.models = gateway
    assert [runtime.retry_delay(None, attempt) for attempt in range(4)] == [0, 0, 0, 0]
    assert runtime.retry_delay(50, 0) == 0.05
    run = runtime.submit(RunCreateRequest(tenant_id="tenant-a", agent_id="event-investigation", input={"message": "分析告警"}))
    final = wait(runtime, run)
    assert final.status == "failed"
    assert final.error_code == "MODEL_TIMEOUT"
    assert gateway.calls == 4  # one attempt plus three retries, not nested retry multiplication
    assert sum(e.event_type == "node.retrying" for e in runtime.store.list_events(run.id)) == 3


def test_dedicated_knowledge_query_is_a_real_intent(runtime):
    decision = runtime.decision.decide({"knowledge_query": "高危告警处理手册"}, runtime.store.get_profile("event-investigation"))
    assert not decision.intent.needs_clarification
    assert decision.intent.params["rewritten_query"] == "高危告警处理手册"


def test_app_lifespan_closes_only_resources_it_owns(runtime):
    owned = create_app()
    with TestClient(owned) as client:
        assert client.get("/health").status_code == 200
    assert owned.state.runtime._closed
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").status_code == 200
    assert not runtime._closed


def test_close_is_idempotent_and_rejects_new_work(runtime):
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.submit(RunCreateRequest(tenant_id="tenant-a", agent_id="event-investigation"))


def test_bundled_profiles_skills_and_workbench_work_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/workbench").status_code == 200
        assert app.state.runtime.store.get_profile("asset-investigation")
        assert app.state.runtime.skills.get("event-investigation")
        assert app.state.runtime.sandbox.root == (tmp_path / "data/opencode-sandbox").resolve()


def test_cli_refuses_unauthenticated_network_listener():
    environment = dict(os.environ, QINGLING_AUTH_ENABLED="false")
    result = subprocess.run([sys.executable, "-m", "app", "--host", "0.0.0.0"], env=environment,
                            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "network binding requires" in result.stderr
