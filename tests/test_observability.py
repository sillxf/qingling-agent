import json
import time

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app
from app.model_gateway import DeterministicModelGateway, ModelGatewayTimeout
from app.runtime import RuntimeService
from app.store import InMemoryStore


def wait_for_terminal(client: TestClient, run_id: str):
    for _ in range(100):
        response = client.get(f"/v1/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed", "cancelled", "timed_out", "waiting_approval"}:
            return body
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


def test_correlation_id_is_propagated_to_run_events_and_audit():
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/v1/runs",
        headers={"X-Correlation-ID": "case:2026-001"},
        json={
            "tenant_id": "tenant-observe",
            "agent_id": "event-investigation",
            "input": {"message": "请分析告警"},
        },
    )
    assert response.status_code == 202
    run = wait_for_terminal(client, response.json()["id"])
    assert run["correlation_id"] == "case:2026-001"
    assert response.headers["x-correlation-id"] == "case:2026-001"

    events = client.get(f"/v1/runs/{run['id']}/events").text
    assert '"correlation_id": "case:2026-001"' in events
    audits = app.state.runtime.store.list_audit()
    assert audits
    assert all(item["correlation_id"] == "case:2026-001" for item in audits)


def test_invalid_correlation_id_is_replaced_and_unknown_run_uses_error_envelope():
    client = TestClient(create_app())
    response = client.post(
        "/v1/runs",
        json={
            "tenant_id": "tenant-observe",
            "agent_id": "event-investigation",
            "correlation_id": "invalid correlation id",
            "input": {"message": "测试"},
        },
    )
    assert response.status_code == 202
    generated = response.json()["correlation_id"]
    assert generated != "invalid correlation id"
    assert generated.isalnum()
    assert len(generated) == 32

    missing = client.get("/v1/runs/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["schema_version"] == "1.0"
    assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_request_validation_uses_safe_error_envelope():
    client = TestClient(create_app())
    response = client.post(
        "/v1/runs",
        headers={"X-Correlation-ID": "validation:001"},
        json={"tenant_id": "tenant-observe", "agent_id": "event-investigation", "input": "must-be-object"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["correlation_id"] == "validation:001"
    assert response.headers["x-correlation-id"] == "validation:001"
    assert "must-be-object" not in response.text


class SlowModelGateway(DeterministicModelGateway):
    def chat(self, **kwargs):
        time.sleep(0.08)
        return super().chat(**kwargs)


def test_run_timeout_is_terminal_and_does_not_publish_output():
    runtime = RuntimeService(
        model_gateway=SlowModelGateway(),
        app_settings=Settings(run_timeout_seconds=0.02, run_workers=1),
    )
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-timeout",
                "agent_id": "event-investigation",
                "input": {"message": "请分析高危攻击告警"},
            },
        )
        assert response.status_code == 202
        run = wait_for_terminal(client, response.json()["id"])
        assert run["status"] == "timed_out"
        assert run["error_code"] == "TIMEOUT"
        assert run["timeout_scope"] == "run"
        assert run["output"] is None
        events = client.get(f"/v1/runs/{run['id']}/events").text
        assert "run.timed_out" in events
        assert "run.completed" not in events
    finally:
        runtime.close()


def test_audit_values_are_redacted_recursively():
    store = InMemoryStore()
    store.append_audit(
        {
            "tenant_id": "tenant-safe",
            "event_type": "tool.called",
            "data": {
                "authorization": "Bearer super-secret-token",
                "nested": {"password": "p@ssword", "message": "api_key=abc123"},
                "email": "analyst@example.com",
                "phone": "13812345678",
            },
        }
    )
    rendered = json.dumps(store.list_audit(), ensure_ascii=False)
    for secret in ("super-secret-token", "p@ssword", "abc123", "analyst@example.com", "13812345678"):
        assert secret not in rendered
    assert "REDACTED" in rendered


def test_approval_cannot_resume_a_timed_out_run():
    runtime = RuntimeService(
        app_settings=Settings(run_timeout_seconds=0.02, run_workers=1),
    )
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-timeout",
                "agent_id": "security-assistant-react",
                "input": {
                    "message": "评估是否需要封禁攻击源",
                    "tool_call": {
                        "tool": "response.block_ip",
                        "args": {"ip": "10.0.0.1", "reason": "confirmed attack"},
                    },
                },
            },
        )
        run = wait_for_terminal(client, response.json()["id"])
        assert run["status"] == "waiting_approval"
        approval_id = next(
            item["data"]["approval_id"]
            for item in runtime.store.list_audit()
            if item["event_type"] == "approval.required"
        )
        time.sleep(0.04)
        decision = client.post(f"/v1/approvals/{approval_id}", json={"approved": True})
        assert decision.status_code == 409
        assert client.get(f"/v1/runs/{run['id']}").json()["status"] == "timed_out"
    finally:
        runtime.close()


class LeakyFailureGateway(DeterministicModelGateway):
    def chat(self, **kwargs):
        raise RuntimeError("api_key=super-secret-token")


def test_runtime_error_is_redacted_in_run_and_sse():
    runtime = RuntimeService(model_gateway=LeakyFailureGateway())
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-safe",
                "agent_id": "event-investigation",
                "input": {"message": "触发错误"},
            },
        )
        run = wait_for_terminal(client, response.json()["id"])
        assert run["status"] == "failed"
        assert run["error_code"] == "INTERNAL"
        assert "super-secret-token" not in run["error"]
        events = client.get(f"/v1/runs/{run['id']}/events").text
        assert "super-secret-token" not in events
        assert "run.error" in events
    finally:
        runtime.close()


class TimeoutFailureGateway(DeterministicModelGateway):
    def chat(self, **kwargs):
        raise ModelGatewayTimeout("upstream model timed out")


def test_model_gateway_error_is_persisted_with_stable_run_contract():
    # Exercise retry exhaustion without coupling assertions to wall-clock backoff.
    runtime = RuntimeService(
        model_gateway=TimeoutFailureGateway(),
        app_settings=Settings(model_gateway_backoff_seconds=0),
    )
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-model",
                "agent_id": "event-investigation",
                "input": {"message": "调用模型"},
            },
        )
        run = wait_for_terminal(client, response.json()["id"])
        assert run["status"] == "failed"
        assert run["error_code"] == "MODEL_TIMEOUT"
        assert run["error_category"] == "timeout"
        assert run["error_retryable"] is True
        events = client.get(f"/v1/runs/{run['id']}/events").text
        assert '"code": "MODEL_TIMEOUT"' in events
    finally:
        runtime.close()


@pytest.mark.parametrize("key", [
    "run_id", "approval_id", "correlation_id", "decision_id", "plan_id", "compensation_id", "audit_id",
])
def test_canonical_uuid_references_survive_recursive_redaction(key):
    from app.observability import redact_value, sanitize_audit_event

    value = "abcdefabcdef4abc8defa13812345678"
    record = {key: value, "data": {key: value}}
    assert redact_value(record) == record
    # Store sanitization is a second pass; references must survive both passes.
    assert sanitize_audit_event(redact_value(record)) == record


def test_uuid_reference_exception_does_not_disable_secret_or_pii_redaction():
    from app.observability import REDACTED, redact_text, redact_value

    value = "abcdefabcdef4abc8defa13812345678"
    assert redact_text(value) != value  # Free-form text retains its existing policy.
    sensitive = {
        "token": value, "api_key": value, "password": value,
        "authorization": "Bearer " + value, "nested": {"refresh_token": value},
    }
    redacted = redact_value(sensitive)
    assert all(redacted[key] == REDACTED for key in ("token", "api_key", "password", "authorization"))
    assert redacted["nested"]["refresh_token"] == REDACTED
    for key in ("approval_id", "run_id", "correlation_id"):
        for text in ("13812345678", "+86 13812345678", "api_key=abc123", "analyst@example.com",
                     "prefix:" + value, value + ":suffix", value.replace("4abc", "0abc")):
            assert redact_value({key: text})[key] != text
    for key in ("id", "customer_id", "tenant_id", "message", "phone", "user_id"):
        assert redact_value({key: value})[key] != value
    assert redact_value({"approval_id": {"password": "secret-value"}}) == {"approval_id": {"password": REDACTED}}
