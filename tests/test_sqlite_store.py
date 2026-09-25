import time
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import AgentProfile, FewShotExample, PortRef, RunCreateRequest, WorkflowEdge, WorkflowManifest, WorkflowNode
from app.runtime import RuntimeService
from app.store import SQLiteStore


def profile() -> AgentProfile:
    return AgentProfile(
        id="sqlite-agent",
        name="SQLite test agent",
        few_shots=[
            FewShotExample(question="q1", answer="a1"),
            FewShotExample(question="q2", answer="a2"),
            FewShotExample(question="q3", answer="a3"),
        ],
    )


def workflow() -> WorkflowManifest:
    return WorkflowManifest(
        id="sqlite-workflow",
        entry="input",
        exit="output",
        nodes=[
            WorkflowNode(id="input", component="input"),
            WorkflowNode(id="output", component="output"),
        ],
        edges=[
            WorkflowEdge(
                from_ref=PortRef(node="input", port="output"),
                to_ref=PortRef(node="output", port="input"),
            )
        ],
    )


def test_sqlite_store_persists_domain_objects_and_recovers_after_reopen(tmp_path):
    db_path = tmp_path / "qingling.db"
    store = SQLiteStore(db_path)
    store.register_profile(profile())
    store.register_workflow(workflow())
    request = RunCreateRequest(
        tenant_id="tenant-a",
        agent_id="sqlite-agent",
        user_id="analyst",
        input={"message": "告警", "token": "do-not-store-in-telemetry"},
        idempotency_key="case-1",
        correlation_id="sqlite:case-1",
    )
    run = store.create_run(
        request,
        profile(),
        correlation_id="sqlite:case-1",
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    event = store.append_event(
        run.id,
        "run.started",
        {"schema_version": "1.0", "tenant_id": "tenant-a", "correlation_id": "sqlite:case-1", "message": "api_key=hidden"},
    )
    approval = store.create_approval(
        run.id,
        "tenant-a",
        "response.block_ip",
        {"ip": "10.0.0.1", "reason": "test"},
        "decision-1",
        "analyst",
        "sqlite:case-1",
    )
    store.append_audit({"tenant_id": "tenant-a", "secret": "super-secret", "run_id": run.id})
    store.close()

    reopened = SQLiteStore(db_path)
    try:
        loaded_profile = reopened.get_profile("sqlite-agent")
        loaded_workflow = reopened.get_workflow("sqlite-workflow")
        loaded_run = reopened.get_run(run.id)
        loaded_events = reopened.list_events(run.id)
        loaded_approval = reopened.get_approval(approval.id)
        assert loaded_profile.id == "sqlite-agent"
        assert loaded_workflow.exit == "output"
        assert loaded_run.correlation_id == "sqlite:case-1"
        assert loaded_run.deadline_at is not None
        assert loaded_events[0].seq == event.seq == 1
        assert loaded_events[0].correlation_id == "sqlite:case-1"
        assert loaded_approval.status == "pending"
        assert "super-secret" not in str(reopened.list_audit())
        assert "REDACTED" in str(reopened.list_audit())
    finally:
        reopened.close()


def test_sqlite_idempotency_is_scoped_to_tenant(tmp_path):
    store = SQLiteStore(tmp_path / "idempotency.db")
    try:
        agent = profile()
        store.register_profile(agent)
        first = store.create_run(
            RunCreateRequest(tenant_id="tenant-a", agent_id=agent.id, input={"message": "one"}, idempotency_key="same"),
            agent,
        )
        repeated = store.create_run(
            RunCreateRequest(tenant_id="tenant-a", agent_id=agent.id, input={"message": "two"}, idempotency_key="same"),
            agent,
        )
        other_tenant = store.create_run(
            RunCreateRequest(tenant_id="tenant-b", agent_id=agent.id, input={"message": "three"}, idempotency_key="same"),
            agent,
        )
        assert repeated.id == first.id
        assert other_tenant.id != first.id
    finally:
        store.close()


def _wait_for_terminal(client: TestClient, run_id: str):
    for _ in range(100):
        response = client.get(f"/v1/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed", "cancelled", "timed_out", "waiting_approval"}:
            return body
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


def test_runtime_sqlite_state_survives_restart(tmp_path):
    db_path = tmp_path / "runtime.db"
    app_settings = Settings(store_backend="sqlite", sqlite_path=str(db_path), run_workers=1)
    first_runtime = RuntimeService(app_settings=app_settings)
    first_run_id = None
    try:
        client = TestClient(create_app(first_runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-restart",
                "agent_id": "event-investigation",
                "user_id": "analyst",
                "input": {"message": "重启前分析告警"},
            },
        )
        assert response.status_code == 202
        first = _wait_for_terminal(client, response.json()["id"])
        assert first["status"] == "succeeded"
        first_run_id = first["id"]
    finally:
        first_runtime.close()

    assert first_run_id is not None
    second_runtime = RuntimeService(app_settings=app_settings)
    try:
        client = TestClient(create_app(second_runtime))
        restored = client.get(f"/v1/runs/{first_run_id}")
        assert restored.status_code == 200
        assert restored.json()["status"] == "succeeded"
        events = second_runtime.store.list_events(first_run_id)
        assert [event.event_type for event in events][:2] == ["run.created", "run.status"]
        assert any(event.event_type == "run.completed" for event in events)

        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-restart",
                "agent_id": "event-investigation",
                "user_id": "analyst",
                "input": {"message": "重启后分析告警"},
            },
        )
        assert response.status_code == 202
        resumed = _wait_for_terminal(client, response.json()["id"])
        assert resumed["status"] == "succeeded"
    finally:
        second_runtime.close()


def test_runtime_sqlite_rebuilds_deadline_for_pending_approval_after_restart(tmp_path):
    db_path = tmp_path / "approval-restart.db"
    app_settings = Settings(
        store_backend="sqlite",
        sqlite_path=str(db_path),
        # This test checks recovery of an expired persisted deadline, not
        # whether CI can finish all SQLite/HTTP setup within 80 milliseconds.
        run_timeout_seconds=30,
        run_workers=1,
    )
    first_runtime = RuntimeService(app_settings=app_settings)
    approval_id = None
    run_id = None
    try:
        client = TestClient(create_app(first_runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-restart-approval",
                "agent_id": "security-assistant-react",
                "input": {
                    "message": "等待审批",
                    "tool_call": {
                        "tool": "response.block_ip",
                        "args": {"ip": "10.0.0.2", "reason": "restart test"},
                    },
                },
            },
        )
        assert response.status_code == 202
        pending = _wait_for_terminal(client, response.json()["id"])
        assert pending["status"] == "waiting_approval"
        run_id = pending["id"]
        approvals = first_runtime.store.list_audit()
        approval_id = next(item["data"]["approval_id"] for item in approvals if item["event_type"] == "approval.required")
        # Simulate downtime deterministically. The restart must reconstruct its
        # monotonic budget from this expired persisted wall-clock deadline.
        persisted = first_runtime.store.get_run(run_id)
        persisted.deadline_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        first_runtime.store.update_run(persisted)
    finally:
        first_runtime.close()

    assert run_id is not None and approval_id is not None
    second_runtime = RuntimeService(app_settings=app_settings)
    try:
        client = TestClient(create_app(second_runtime))
        decision = client.post(f"/v1/approvals/{approval_id}", json={"approved": True})
        assert decision.status_code == 409
        restored = client.get(f"/v1/runs/{run_id}").json()
        assert restored["status"] == "timed_out"
    finally:
        second_runtime.close()
