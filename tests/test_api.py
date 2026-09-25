import time

from fastapi.testclient import TestClient

from app.main import create_app


def wait_for_terminal(client: TestClient, run_id: str):
    for _ in range(100):
        response = client.get(f"/v1/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed", "cancelled", "timed_out", "waiting_approval"}:
            return body
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


def test_health_and_workflow_run():
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["service"] == "青灵智能体"

    response = client.post(
        "/v1/runs",
        json={
            "tenant_id": "tenant-a",
            "agent_id": "event-investigation",
            "user_id": "analyst",
            "input": {"message": "请分析一条高危攻击告警"},
            "idempotency_key": "case-001",
        },
    )
    assert response.status_code == 202
    run = wait_for_terminal(client, response.json()["id"])
    assert run["status"] == "succeeded"
    assert "report" in run["output"]

    repeated = client.post(
        "/v1/runs",
        json={
            "tenant_id": "tenant-a",
            "agent_id": "event-investigation",
            "user_id": "analyst",
            "input": {"message": "重复提交"},
            "idempotency_key": "case-001",
        },
    )
    assert repeated.status_code == 202
    assert repeated.json()["id"] == run["id"]


def test_react_high_risk_tool_waits_for_approval():
    client = TestClient(create_app())
    response = client.post(
        "/v1/runs",
        json={
            "tenant_id": "tenant-a",
            "agent_id": "security-assistant-react",
            "user_id": "analyst",
            "input": {
                "message": "评估是否需要封禁攻击源",
                "tool_call": {"tool": "response.block_ip", "args": {"ip": "10.0.0.1", "reason": "confirmed attack"}},
            },
        },
    )
    assert response.status_code == 202
    run = wait_for_terminal(client, response.json()["id"])
    assert run["status"] == "waiting_approval"
    events = client.get(f"/v1/runs/{run['id']}/events").text
    assert "approval.required" in events
