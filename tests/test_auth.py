import json
import time

import pytest
from fastapi.testclient import TestClient

from app.auth import AuthConfigurationError, StaticTokenAuthenticator
from app.config import Settings
from app.main import create_app
from app.runtime import RuntimeService


def auth_settings() -> Settings:
    return Settings(
        auth_enabled=True,
        auth_tokens_json=json.dumps(
            [
                {
                    "token": "analyst-token-123",
                    "subject": "analyst-a",
                    "tenant_id": "tenant-a",
                    "roles": ["analyst"],
                    "token_id": "analyst-a-1",
                },
                {
                    "token": "operator-token-456",
                    "subject": "operator-a",
                    "tenant_id": "tenant-a",
                    "roles": ["operator"],
                    "token_id": "operator-a-1",
                },
                {
                    "token": "operator-token-789",
                    "subject": "operator-b",
                    "tenant_id": "tenant-b",
                    "roles": ["operator"],
                    "token_id": "operator-b-1",
                },
            ]
        ),
    )


def headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def wait_for_terminal(client: TestClient, run_id: str, token: str) -> dict:
    for _ in range(100):
        response = client.get(f"/v1/runs/{run_id}", headers=headers(token))
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed", "cancelled", "timed_out", "waiting_approval"}:
            return body
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


def test_authentication_is_required_when_enabled():
    runtime = RuntimeService(app_settings=auth_settings())
    try:
        client = TestClient(create_app(runtime))
        response = client.get("/v1/tools")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHENTICATED"

        invalid = client.get("/v1/tools", headers=headers("wrong-token"))
        assert invalid.status_code == 401
        assert invalid.json()["error"]["code"] == "UNAUTHENTICATED"
    finally:
        runtime.close()


def test_tenant_denial_returns_normalized_correlation_id():
    runtime = RuntimeService(app_settings=auth_settings())
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            headers={**headers("operator-token-789"), "X-Correlation-ID": "bad correlation id"},
            json={
                "tenant_id": "tenant-a",
                "agent_id": "event-investigation",
                "input": {"message": "越权"},
            },
        )
        assert response.status_code == 403
        body = response.json()
        assert body["error"]["code"] == "FORBIDDEN"
        assert body["correlation_id"] != "bad correlation id"
        assert response.headers["x-correlation-id"] == body["correlation_id"]
    finally:
        runtime.close()


def test_principal_controls_user_and_tenant_access():
    runtime = RuntimeService(app_settings=auth_settings())
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            headers=headers("analyst-token-123"),
            json={
                "tenant_id": "tenant-a",
                "agent_id": "event-investigation",
                "user_id": "spoofed-user",
                "input": {"message": "分析告警"},
            },
        )
        assert response.status_code == 202
        run = wait_for_terminal(client, response.json()["id"], "analyst-token-123")
        assert run["user_id"] == "analyst-a"

        cross_tenant = client.get(f"/v1/runs/{run['id']}", headers=headers("operator-token-789"))
        assert cross_tenant.status_code == 403
        assert cross_tenant.json()["error"]["code"] == "FORBIDDEN"
    finally:
        runtime.close()


def test_management_roles_and_approval_roles_are_enforced():
    runtime = RuntimeService(app_settings=auth_settings())
    try:
        client = TestClient(create_app(runtime))
        profile_payload = {
            "id": "auth-test-profile",
            "name": "Auth test profile",
            "few_shots": [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
                {"question": "q3", "answer": "a3"},
            ],
        }
        denied = client.post("/v1/agents/profile", headers=headers("analyst-token-123"), json=profile_payload)
        assert denied.status_code == 403

        allowed = client.post("/v1/agents/profile", headers=headers("operator-token-456"), json=profile_payload)
        assert allowed.status_code == 200

        response = client.post(
            "/v1/runs",
            headers=headers("analyst-token-123"),
            json={
                "tenant_id": "tenant-a",
                "agent_id": "security-assistant-react",
                "input": {
                    "message": "评估封禁",
                    "tool_call": {
                        "tool": "response.block_ip",
                        "args": {"ip": "10.0.0.9", "reason": "auth test"},
                    },
                },
            },
        )
        pending = wait_for_terminal(client, response.json()["id"], "analyst-token-123")
        assert pending["status"] == "waiting_approval"
        approval_id = next(
            item["data"]["approval_id"]
            for item in runtime.store.list_audit()
            if item["event_type"] == "approval.required"
        )

        analyst_decision = client.post(
            f"/v1/approvals/{approval_id}",
            headers=headers("analyst-token-123"),
            json={"approved": True},
        )
        assert analyst_decision.status_code == 403

        operator_decision = client.post(
            f"/v1/approvals/{approval_id}",
            headers=headers("operator-token-456"),
            json={"approved": True},
        )
        assert operator_decision.status_code == 200
    finally:
        runtime.close()


def test_auth_configuration_is_validated_and_tokens_are_not_returned():
    with pytest.raises(AuthConfigurationError):
        StaticTokenAuthenticator.from_json("")
    with pytest.raises(AuthConfigurationError):
        StaticTokenAuthenticator.from_json("not-json")
    with pytest.raises(AuthConfigurationError):
        StaticTokenAuthenticator.from_json(
            json.dumps([{"token": "short", "subject": "user", "tenant_id": "tenant", "roles": []}])
        )
