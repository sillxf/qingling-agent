from __future__ import annotations

import json
import threading
import time

import pytest

from app.config import Settings
from app.model_gateway import (
    DeterministicModelGateway,
    ModelGatewayConfigurationError,
    ModelCallContext,
    ModelGatewayCancelled,
    ModelGatewayHTTPError,
    ModelGatewayResponseError,
    ModelGatewayTimeout,
    OpenAICompatibleModelGateway,
    build_model_gateway,
)


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


def gateway(opener: SequenceOpener, **kwargs) -> OpenAICompatibleModelGateway:
    options = {
        "base_url": "https://models.example.test/v1",
        "api_key": "test-key",
        "opener": opener,
        "sleep": lambda _: None,
    }
    options.update(kwargs)
    return OpenAICompatibleModelGateway(**options)


def test_deterministic_gateway_contract_is_unchanged() -> None:
    adapter = DeterministicModelGateway()
    response = adapter.chat(
        model="sec-llm",
        system_prompt="system",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.1,
        max_tokens=32,
    )
    assert response["reason_code"] == "offline_demo"
    assert response["model"] == "sec-llm"
    assert len(adapter.embed(model="bge-m3", texts=["a", "b"])) == 2


def test_chat_maps_openai_request_and_response() -> None:
    opener = SequenceOpener(
        FakeResponse(
            200,
            {
                "id": "chat-1",
                "model": "sec-llm-v2",
                "choices": [{"message": {"role": "assistant", "content": "分析完成"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )
    )
    adapter = gateway(opener, timeout_seconds=3.5, max_retries=0)

    result = adapter.chat(
        model="sec-llm",
        system_prompt="你是安全分析师",
        messages=[{"role": "user", "content": "请分析"}],
        temperature=0.2,
        max_tokens=128,
    )

    request, timeout = opener.calls[0]
    assert timeout == 3.5
    assert request.full_url == "https://models.example.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type") == "application/json"
    body = json.loads(request.data.decode("utf-8"))
    assert body == {
        "model": "sec-llm",
        "messages": [
            {"role": "system", "content": "你是安全分析师"},
            {"role": "user", "content": "请分析"},
        ],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    assert result == {
        "type": "final",
        "content": "分析完成",
        "model": "sec-llm-v2",
        "reason_code": "stop",
        "id": "chat-1",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


def test_embeddings_are_sorted_by_index_and_empty_input_is_local() -> None:
    opener = SequenceOpener(
        FakeResponse(
            200,
            {
                "data": [
                    {"index": 1, "embedding": [2, 3]},
                    {"index": 0, "embedding": [0, 1]},
                ]
            },
        )
    )
    adapter = gateway(opener, max_retries=0)
    assert adapter.embed(model="bge-m3", texts=["first", "second"]) == [[0.0, 1.0], [2.0, 3.0]]
    assert adapter.embed(model="bge-m3", texts=[]) == []
    assert len(opener.calls) == 1
    request, _ = opener.calls[0]
    assert request.full_url.endswith("/embeddings")


def test_timeout_and_server_errors_retry_with_exponential_backoff() -> None:
    opener = SequenceOpener(
        TimeoutError("slow"),
        FakeResponse(500, {"error": {"message": "temporary"}}),
        FakeResponse(200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}),
    )
    delays: list[float] = []
    adapter = OpenAICompatibleModelGateway(
        base_url="https://models.example.test/v1",
        opener=opener,
        timeout_seconds=1,
        max_retries=2,
        backoff_seconds=0.25,
        sleep=delays.append,
    )

    result = adapter.chat(
        model="m",
        system_prompt="",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
        max_tokens=16,
    )
    assert result["content"] == "ok"
    assert len(opener.calls) == 3
    assert delays == [0.25, 0.5]


def test_retry_after_header_overrides_backoff_and_http_4xx_is_not_retryable() -> None:
    opener = SequenceOpener(
        FakeResponse(429, {"error": {"message": "rate limited"}}, {"Retry-After": "0.4"}),
        FakeResponse(401, {"error": {"message": "api_key=not-leaked"}}),
    )
    delays: list[float] = []
    adapter = gateway(opener, max_retries=1, backoff_seconds=9, sleep=delays.append)

    with pytest.raises(ModelGatewayHTTPError) as raised:
        adapter.chat(
            model="m",
            system_prompt="",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=16,
        )
    error = raised.value
    assert error.status_code == 401
    assert error.retryable is False
    assert error.retry_after_ms is None
    assert "not-leaked" not in str(error)
    assert delays == [0.4]
    assert len(opener.calls) == 2


def test_malformed_response_is_classified_without_retry() -> None:
    opener = SequenceOpener(FakeResponse(200, {"choices": []}))
    adapter = gateway(opener, max_retries=3, backoff_seconds=0)
    with pytest.raises(ModelGatewayResponseError) as raised:
        adapter.chat(
            model="m",
            system_prompt="",
            messages=[],
            temperature=0,
            max_tokens=16,
        )
    assert raised.value.retryable is False
    assert len(opener.calls) == 1


def test_gateway_configuration_is_validated_and_settings_are_supported() -> None:
    app_settings = Settings(
        model_gateway_base_url="https://models.example.test/v1",
        model_gateway_api_key="settings-key",
        model_gateway_timeout_seconds=2.5,
        model_gateway_max_retries=1,
        model_gateway_backoff_seconds=0.1,
    )
    adapter = OpenAICompatibleModelGateway(app_settings=app_settings, opener=SequenceOpener())
    assert adapter.base_url == "https://models.example.test/v1"
    assert adapter.api_key == "settings-key"
    assert adapter.timeout_seconds == 2.5
    assert adapter.max_retries == 1
    assert adapter.backoff_seconds == 0.1

    with pytest.raises(ModelGatewayConfigurationError):
        OpenAICompatibleModelGateway(base_url="")
    with pytest.raises(ModelGatewayConfigurationError):
        OpenAICompatibleModelGateway(base_url="file:///tmp/models")
    with pytest.raises(ModelGatewayConfigurationError):
        OpenAICompatibleModelGateway(base_url="https://models.example.test", timeout_seconds=0)
    with pytest.raises(ModelGatewayConfigurationError):
        OpenAICompatibleModelGateway(base_url="https://models.example.test", max_retries=11)


def test_timeout_error_exposes_retry_contract() -> None:
    opener = SequenceOpener(TimeoutError("slow"))
    adapter = gateway(opener, max_retries=0)
    with pytest.raises(ModelGatewayTimeout) as raised:
        adapter.chat(
            model="m",
            system_prompt="",
            messages=[],
            temperature=0,
            max_tokens=16,
        )
    assert raised.value.code == "MODEL_TIMEOUT"
    assert raised.value.category == "timeout"
    assert raised.value.retryable is True


def test_model_gateway_factory_is_offline_by_default_and_remote_when_configured() -> None:
    offline = build_model_gateway(Settings(model_gateway_base_url=""))
    assert isinstance(offline, DeterministicModelGateway)

    remote = build_model_gateway(Settings(model_gateway_base_url="https://models.example.test/v1"))
    assert isinstance(remote, OpenAICompatibleModelGateway)
    assert remote.base_url == "https://models.example.test/v1"


def test_context_budget_is_forwarded_to_http_and_cancel_is_fail_closed() -> None:
    opener = SequenceOpener(FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}))
    adapter = gateway(opener, timeout_seconds=30, max_retries=0)
    result = adapter.chat_with_context(
        model="m",
        system_prompt="",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
        max_tokens=16,
        context=ModelCallContext(timeout_seconds=1.25),
    )
    assert result["content"] == "ok"
    assert 0 < opener.calls[0][1] <= 1.25

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ModelGatewayCancelled):
        adapter.chat_with_context(
            model="m",
            system_prompt="",
            messages=[],
            temperature=0,
            max_tokens=16,
            context=ModelCallContext(cancel_event=cancelled),
        )
    assert len(opener.calls) == 1


def test_expired_context_deadline_does_not_start_a_provider_request() -> None:
    opener = SequenceOpener(FakeResponse(200, {"choices": [{"message": {"content": "late"}}]}))
    adapter = gateway(opener, max_retries=2)
    with pytest.raises(ModelGatewayTimeout):
        adapter.chat_with_context(
            model="m",
            system_prompt="",
            messages=[],
            temperature=0,
            max_tokens=16,
            context=ModelCallContext(deadline_monotonic=time.monotonic() - 1),
        )
    assert opener.calls == []
