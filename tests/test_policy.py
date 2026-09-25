import time

from fastapi.testclient import TestClient

from app.models import AgentProfile, FewShotExample, ToolContract
from app.policy import PolicyGate
from app.main import create_app
from app.runtime import RuntimeService


def profile(**permissions):
    return AgentProfile(
        id="test",
        name="test",
        permissions=permissions,
        few_shots=[
            FewShotExample(question="q1", answer="a1"),
            FewShotExample(question="q2", answer="a2"),
            FewShotExample(question="q3", answer="a3"),
        ],
    )


def test_policy_allows_valid_read_call():
    tool = ToolContract(
        name="event.search",
        permission="read",
        input_schema={"required": ["query"], "properties": {"query": {"type": "string"}}},
    )
    decision = PolicyGate().evaluate(profile(read="allow"), tool, {"query": "ip:10.0.0.1"})
    assert decision.allowed is True
    assert decision.requires_approval is False


def test_policy_requires_approval_for_high_risk_call():
    tool = ToolContract(
        name="response.block_ip",
        permission="mutate",
        risk_level="critical",
        requires_approval=True,
        input_schema={"required": ["ip"], "properties": {"ip": {"type": "string"}}},
    )
    decision = PolicyGate().evaluate(profile(mutate="allow"), tool, {"ip": "10.0.0.1"})
    assert decision.allowed is False
    assert decision.requires_approval is True


def test_policy_rejects_invalid_arguments():
    tool = ToolContract(
        name="event.get",
        permission="read",
        input_schema={"required": ["event_id"], "properties": {"event_id": {"type": "string"}}},
    )
    decision = PolicyGate().evaluate(profile(read="allow"), tool, {})
    assert decision.allowed is False
    assert "missing required" in decision.reason


def test_react_rejects_non_object_tool_arguments():
    runtime = RuntimeService()
    try:
        http = TestClient(create_app(runtime))
        response = http.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-a",
                "agent_id": "security-assistant-react",
                "input": {
                    "message": "invalid args",
                    "tool_call": {"tool": "event.search", "args": "not-an-object"},
                },
            },
        )
        run_id = response.json()["id"]
        for _ in range(100):
            body = http.get(f"/v1/runs/{run_id}").json()
            if body["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert body["status"] == "failed"
        assert body["error_code"] == "POLICY_DENIED"
    finally:
        runtime.close()
