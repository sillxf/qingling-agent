import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.event_source import (
    DemoEventSource,
    EventSourceConfigurationError,
    EventSourceHTTPError,
    EventSourceResponseError,
    EventSourceTimeout,
    HTTPEventSource,
)
from app.main import create_app
from app.runtime import RuntimeService
from app.tools import ToolInvocationContext, build_default_tools


class FakeResponse:
    def __init__(self, status: int, payload: object, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


class SequenceOpener:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request: object, *, timeout: float):
        self.calls.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def context(**kwargs) -> ToolInvocationContext:
    values = {
        "run_id": "run-1",
        "tenant_id": "tenant-a",
        "user_id": "analyst-a",
        "correlation_id": "corr-1",
    }
    values.update(kwargs)
    return ToolInvocationContext(**values)


def http_source(opener: SequenceOpener, **kwargs) -> HTTPEventSource:
    options = {
        "base_url": "https://events.example.test/api",
        "api_key": "event-key",
        "opener": opener,
        "max_retries": 0,
    }
    options.update(kwargs)
    return HTTPEventSource(**options)


def test_demo_event_source_marks_results_as_simulated():
    source = DemoEventSource()
    assert source.search(query="ip:10.0.0.1", context=context())["simulated"] is True
    assert source.get(event_id="evt-1", context=context())["source"] == "demo-event-source"


def test_default_demo_tools_keep_direct_invoke_compatibility():
    registry = build_default_tools(Settings(event_api_base_url=""))
    result = registry.invoke("event.search", {"query": "ip:10.0.0.1"})
    assert result["simulated"] is True


def test_remote_event_tools_reject_direct_invoke_without_context():
    source = http_source(SequenceOpener(FakeResponse(200, {"items": [], "total": 0})))
    registry = build_default_tools(Settings(event_api_base_url="https://events.example.test/api"), event_source=source)
    with pytest.raises(ValueError, match="trusted tool context"):
        registry.invoke("event.search", {"query": "x"})


def test_http_event_search_forwards_trusted_scope_headers():
    opener = SequenceOpener(FakeResponse(200, {"items": [{"id": "evt-1"}], "total": 1, "source": "siem"}))
    source = http_source(opener)
    result = source.search(query="ip:10.0.0.1", context=context())
    request, timeout = opener.calls[0]
    assert request.full_url == "https://events.example.test/api/events/search"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer event-key"
    assert request.headers["X-tenant-id"] == "tenant-a"
    assert request.headers["X-correlation-id"] == "corr-1"
    assert request.headers["X-run-id"] == "run-1"
    assert json.loads(request.data.decode("utf-8")) == {"query": "ip:10.0.0.1"}
    assert timeout > 0
    assert result == {
        "query": "ip:10.0.0.1",
        "items": [{"id": "evt-1"}],
        "total": 1,
        "source": "siem",
        "simulated": False,
    }


def test_http_event_get_encodes_identifier_and_normalizes_result():
    opener = SequenceOpener(FakeResponse(200, {"event": {"id": "a/b", "severity": "high"}}))
    source = http_source(opener)
    result = source.get(event_id="a/b?secret", context=context())
    request, _ = opener.calls[0]
    assert request.full_url.endswith("/events/a%2Fb%3Fsecret")
    assert request.get_method() == "GET"
    assert result["event"] == {"id": "a/b", "severity": "high"}
    assert result["simulated"] is False


def test_http_event_retries_transient_errors_and_classifies_bad_payloads():
    delays: list[float] = []
    opener = SequenceOpener(
        FakeResponse(503, {"error": {"message": "temporary api_key=hidden"}}),
        FakeResponse(200, {"items": [], "total": 0}),
    )
    source = http_source(opener, max_retries=1, backoff_seconds=0.3, sleep=delays.append)
    assert source.search(query="x", context=context())["total"] == 0
    assert delays == [0.3]

    malformed = http_source(SequenceOpener(FakeResponse(200, {"items": "bad"})))
    with pytest.raises(EventSourceResponseError):
        malformed.search(query="x", context=context())

    timeout_source = http_source(SequenceOpener(TimeoutError("slow")))
    with pytest.raises(EventSourceTimeout):
        timeout_source.search(query="x", context=context())


def test_http_event_provider_error_does_not_leak_credentials():
    source = http_source(
        SequenceOpener(FakeResponse(401, {"error": {"message": "api_key=secret-value"}}))
    )
    with pytest.raises(EventSourceHTTPError) as raised:
        source.search(query="x", context=context())
    assert "secret-value" not in str(raised.value)
    assert raised.value.status_code == 401


def test_http_event_rejects_control_characters_in_trusted_context():
    source = http_source(SequenceOpener(FakeResponse(200, {"items": [], "total": 0})))
    with pytest.raises(EventSourceConfigurationError) as raised:
        source.search(query="x", context=context(tenant_id="tenant-a\r\nX-Injected: yes"))
    assert "invalid" in str(raised.value)


def test_react_allowed_read_tool_executes_with_context_and_source_metadata():
    runtime = RuntimeService()
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-a",
                "agent_id": "security-assistant-react",
                "input": {
                    "message": "查询告警",
                    "tool_call": {"tool": "event.search", "args": {"query": "ip:10.0.0.1"}},
                },
            },
        )
        assert response.status_code == 202
        for _ in range(100):
            body = client.get(f"/v1/runs/{response.json()['id']}").json()
            if body["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert body["status"] == "succeeded"
        assert body["output"]["tool_observation"]["simulated"] is True
        assert "tool.completed" in client.get(f"/v1/runs/{body['id']}/events").text
    finally:
        runtime.close()


def test_event_source_failure_maps_to_run_error_code():
    class FailingSource(DemoEventSource):
        def search(self, *, query, context):
            raise EventSourceTimeout("event backend timed out")

    settings = Settings(run_timeout_seconds=2)
    runtime = RuntimeService(
        app_settings=settings,
        tool_registry=build_default_tools(settings, event_source=FailingSource()),
    )
    try:
        client = TestClient(create_app(runtime))
        response = client.post(
            "/v1/runs",
            json={
                "tenant_id": "tenant-a",
                "agent_id": "security-assistant-react",
                "input": {
                    "message": "查询告警",
                    "tool_call": {"tool": "event.search", "args": {"query": "x"}},
                },
            },
        )
        run_id = response.json()["id"]
        for _ in range(100):
            body = client.get(f"/v1/runs/{run_id}").json()
            if body["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert body["status"] == "failed"
        assert body["error_code"] == "EVENT_SOURCE_TIMEOUT"
        assert body["error_category"] == "timeout"
        assert body["error_retryable"] is True
    finally:
        runtime.close()
