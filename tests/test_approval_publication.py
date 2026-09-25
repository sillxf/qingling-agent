"""Deterministic concurrency regressions for approval publication and SSE tails."""
from threading import Event

from fastapi.testclient import TestClient
import pytest

from app.bootstrap import load_demo_configuration
from app.config import Settings
from app.main import create_app
from app.models import RunCreateRequest, WorkflowManifest
from app.runtime import RuntimeService
from app.runtime_errors import ApprovalRequired
from app.store import InMemoryStore, SQLiteStore


@pytest.fixture(params=["memory", "sqlite"])
def runtime(request, tmp_path):
    store = InMemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "publication.db")
    service = RuntimeService(store=store, app_settings=Settings(
        run_workers=1, run_timeout_seconds=30, store_backend="memory", knowledge_store_backend="memory",
        opencode_sandbox_root=str(tmp_path / "sandbox"),
    ))
    load_demo_configuration(service)
    yield service
    service.close()


def start_approval(runtime, engine):
    if engine == "react":
        return runtime.submit(RunCreateRequest(
            tenant_id="tenant-a", agent_id="security-assistant-react",
            input={"tool_call": {"tool": "response.block_ip", "args": {
                "ip": "192.0.2.10", "reason": "synthetic publication regression"}}},
        ))
    workflow = WorkflowManifest(
        id="publication-gate", entry="gate", exit="gate", edges=[],
        nodes=[{"id": "gate", "component": "control.approval", "bindings": {"input": {"value": {}}}}],
    )
    runtime.publish_workflow(workflow)
    return runtime.submit(RunCreateRequest(tenant_id="tenant-a", agent_id="event-investigation"), workflow=workflow)


@pytest.mark.parametrize("engine", ["react", "workflow"])
def test_waiting_status_is_not_visible_before_approval_audit(runtime, monkeypatch, engine):
    audit_entered, release_audit = Event(), Event()
    original = runtime.store.append_audit

    def gated_audit(item):
        if item["event_type"] == "approval.required":
            audit_entered.set()
            if not release_audit.wait(5):
                raise AssertionError("test did not release the audit writer")
        original(item)

    monkeypatch.setattr(runtime.store, "append_audit", gated_audit)
    run = start_approval(runtime, engine)
    try:
        assert audit_entered.wait(5), "execution did not reach approval publication"
        # Inspect the committed store directly, without relying on a race in GET.
        # The writer is paused at an exact point, not slowed by a random sleep.
        visible = runtime.store.get_run(run.id)
        assert visible.status == "running", "waiting_approval was published before its audit"
        assert not any(item["event_type"] == "approval.required" for item in runtime.store.list_audit())
    finally:
        release_audit.set()
        runtime._futures[run.id].result(timeout=5)

    with TestClient(create_app(runtime)) as client:
        pending = client.get(f"/v1/runs/{run.id}").json()
        assert pending["status"] == "waiting_approval"
        audits = [item for item in runtime.store.list_audit() if item["event_type"] == "approval.required"]
        assert len(audits) == 1
        approval_id = audits[0]["data"]["approval_id"]
        events = runtime.store.list_events(run.id)
        required = next(event for event in events if event.event_type == "approval.required")
        assert audits[0]["event_seq"] == required.seq
        assert sum(event.event_type == "approval.required" for event in events) == 1
        assert "event: approval.required" in client.get(f"/v1/runs/{run.id}/events").text
        decision = client.post(f"/v1/approvals/{approval_id}", json={"approved": True})
        assert decision.status_code == 200
        runtime._futures[run.id].result(timeout=5)
        assert runtime.get_run(run.id).status == "succeeded"
        assert runtime.store.get_approval(approval_id).consumed_at is not None
        assert sum(event.event_type == "approval.required" for event in runtime.store.list_events(run.id)) == 1


@pytest.mark.parametrize("engine", ["react", "workflow"])
@pytest.mark.parametrize("approved", [True, False])
def test_decision_during_original_worker_unwind(runtime, monkeypatch, engine, approved):
    """A visible pause remains actionable before its original worker returns."""
    published, release = Event(), Event()
    original_request = runtime.request_approval

    def hold_after_publication(approval):
        try:
            original_request(approval)
        except ApprovalRequired:
            published.set()
            if not release.wait(5):
                raise AssertionError("test did not release original worker")
            raise

    monkeypatch.setattr(runtime, "request_approval", hold_after_publication)
    run = start_approval(runtime, engine)
    original_future = runtime._futures[run.id]
    try:
        assert published.wait(5), "approval pause was not published"
        with TestClient(create_app(runtime)) as client:
            visible = client.get(f"/v1/runs/{run.id}").json()
            assert visible["status"] == "waiting_approval"
            approval_id = next(event.data["approval_id"] for event in runtime.store.list_events(run.id)
                               if event.event_type == "approval.required")
            response = client.post(f"/v1/approvals/{approval_id}", json={"approved": approved})
            assert response.status_code == 200, response.text
            assert runtime.store.get_approval(approval_id).status == ("approved" if approved else "rejected")
    finally:
        release.set()
    original_future.result(timeout=5)
    if approved:
        runtime._futures[run.id].result(timeout=5)
    assert runtime.get_run(run.id).status == ("succeeded" if approved else "failed")


@pytest.mark.parametrize("final_status,tail", [("waiting_approval", "approval.required"), ("succeeded", "run.completed")])
def test_sse_does_not_drop_tail_published_after_its_event_read(runtime, monkeypatch, final_status, tail):
    request = RunCreateRequest(tenant_id="tenant-a", agent_id="event-investigation")
    run = runtime.store.create_run(request, runtime.store.get_profile(request.agent_id))
    runtime.transition(run.id, "running")
    original = runtime.store.list_events
    injected = False

    def publish_after_snapshot(run_id, after_seq=0):
        nonlocal injected
        previous_events = original(run_id, after_seq)
        if run_id == run.id and not injected:
            injected = True
            # Force precisely the old list-events -> publish -> read-status
            # interleaving. A stream must not treat a newer settled status as
            # proof that its earlier event list already contains the tail.
            runtime.emit(run.id, tail, {"approval_id": "synthetic-tail"})
            runtime.transition(run.id, final_status)
        return previous_events

    monkeypatch.setattr(runtime.store, "list_events", publish_after_snapshot)
    with TestClient(create_app(runtime)) as client:
        response = client.get(f"/v1/runs/{run.id}/events")
        assert response.status_code == 200
        assert injected
        assert response.text.count(f"event: {tail}\n") == 1
        last = original(run.id)[-1].seq
        replay = client.get(f"/v1/runs/{run.id}/events", headers={"Last-Event-ID": str(last)})
        assert replay.status_code == 200 and replay.text == ""


@pytest.mark.parametrize("engine", ["react", "workflow"])
def test_cancellation_before_pause_does_not_publish_actionable_approval(runtime, monkeypatch, engine):
    original = runtime.store.create_approval

    def cancel_after_creation(*args, **kwargs):
        approval = original(*args, **kwargs)
        runtime.cancel(approval.run_id)
        return approval

    monkeypatch.setattr(runtime.store, "create_approval", cancel_after_creation)
    run = start_approval(runtime, engine)
    runtime._futures[run.id].result(timeout=5)
    assert runtime.get_run(run.id).status == "cancelled"
    assert not any(event.event_type == "approval.required" for event in runtime.store.list_events(run.id))


@pytest.mark.parametrize("engine", ["react", "workflow"])
@pytest.mark.parametrize("approved", [True, False])
def test_phone_like_approval_id_remains_actionable(runtime, phone_like_uuids, engine, approved):
    """Telemetry must not turn a persisted approval ID into a different ID."""
    run = start_approval(runtime, engine)
    runtime._futures[run.id].result(timeout=5)
    assert runtime.get_run(run.id).status == "waiting_approval"
    required = next(event for event in runtime.store.list_events(run.id)
                    if event.event_type == "approval.required")
    approval_id = required.data["approval_id"]
    with TestClient(create_app(runtime)) as client:
        response = client.post(f"/v1/approvals/{approval_id}", json={"approved": approved})
        assert response.status_code == 200, response.text
        assert response.json()["id"] == approval_id
        stream = client.get(f"/v1/runs/{run.id}/events").text
        assert approval_id in stream
    if approved:
        runtime._futures[run.id].result(timeout=5)
    assert runtime.get_run(run.id).status == ("succeeded" if approved else "failed")
    approval = runtime.store.get_approval(approval_id)
    assert approval.run_id == run.id
    assert approval.consumed_at is not None if approved else approval.consumed_at is None
    audit = next(item for item in runtime.store.list_audit() if item["event_type"] == "approval.required")
    assert required.run_id == required.data["run_id"] == audit["run_id"] == audit["data"]["run_id"] == run.id
    assert audit["data"]["approval_id"] == approval_id
