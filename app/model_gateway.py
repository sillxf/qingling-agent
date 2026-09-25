from __future__ import annotations

import json
import math
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import Settings, settings
from .observability import redact_text


class ModelGatewayError(RuntimeError):
    """Base error for model calls.

    The fields intentionally mirror the Runtime error contract so callers can
    classify failures without parsing provider-specific exception messages.
    """

    code = "MODEL_GATEWAY_ERROR"
    category = "model"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        retryable: Optional[bool] = None,
        retry_after_ms: Optional[int] = None,
        status_code: Optional[int] = None,
    ) -> None:
        self.retryable = self.__class__.retryable if retryable is None else bool(retryable)
        self.retry_after_ms = retry_after_ms
        self.status_code = status_code
        super().__init__(message)


class ModelGatewayConfigurationError(ModelGatewayError):
    code = "MODEL_CONFIGURATION_ERROR"
    category = "configuration"


class ModelGatewayTimeout(ModelGatewayError):
    code = "MODEL_TIMEOUT"
    category = "timeout"
    retryable = True


class ModelGatewayTransportError(ModelGatewayError):
    code = "MODEL_TRANSPORT_ERROR"
    category = "transport"
    retryable = True


class ModelGatewayHTTPError(ModelGatewayError):
    code = "MODEL_HTTP_ERROR"
    category = "provider"

    def __init__(self, status_code: int, message: str, *, retry_after_ms: Optional[int] = None) -> None:
        self.status_code = status_code
        super().__init__(
            message,
            retryable=_http_status_retryable(status_code),
            retry_after_ms=retry_after_ms,
            status_code=status_code,
        )


class ModelGatewayResponseError(ModelGatewayError):
    code = "MODEL_RESPONSE_ERROR"
    category = "response"


class ModelGatewayCancelled(ModelGatewayError):
    code = "MODEL_CANCELLED"
    category = "execution"


# Descriptive aliases keep integrations free to use either the provider or
# transport-oriented terminology.
ModelGatewayTimeoutError = ModelGatewayTimeout
ModelGatewayHTTPStatusError = ModelGatewayHTTPError


def _http_status_retryable(status_code: int) -> bool:
    return status_code in {408, 425, 429} or status_code >= 500


class ModelGateway(ABC):
    @abstractmethod
    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def chat_with_context(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        context: Optional["ModelCallContext"] = None,
    ) -> Dict[str, Any]:
        """Call chat while preserving compatibility with simple adapters.

        Adapters that understand budgets or cancellation can override this
        method. The default delegates to the original ``chat`` contract, so
        existing injected gateways do not need to change immediately.
        """

        return self.chat(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @abstractmethod
    def embed(self, *, model: str, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_with_context(
        self,
        *,
        model: str,
        texts: List[str],
        context: Optional["ModelCallContext"] = None,
    ) -> List[List[float]]:
        return self.embed(model=model, texts=texts)


@dataclass(frozen=True)
class ModelCallContext:
    """Execution budget passed from RuntimeService to a model adapter."""

    timeout_seconds: Optional[float] = None
    deadline_monotonic: Optional[float] = None
    cancel_event: Optional[threading.Event] = None

    def remaining_seconds(self, default: float) -> float:
        values = [float(default)]
        if self.timeout_seconds is not None:
            values.append(max(0.0, float(self.timeout_seconds)))
        if self.deadline_monotonic is not None:
            values.append(max(0.0, self.deadline_monotonic - time.monotonic()))
        return min(values)

    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()


class DeterministicModelGateway(ModelGateway):
    """Offline-safe adapter used until a real OpenAI-compatible endpoint is configured."""

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        user_message = messages[-1]["content"] if messages else ""
        return {
            "type": "final",
            "content": (
                "青灵智能体已接收任务。当前运行在离线演示模式，真实 Sec LLM 尚未接入；"
                "系统已完成任务封装、流程校验、策略检查和审计记录。"
            ),
            "model": model,
            "input_preview": user_message[:160],
            "reason_code": "offline_demo",
        }

    def embed(self, *, model: str, texts: List[str]) -> List[List[float]]:
        # Stable placeholder vectors keep the RAG contract testable offline.
        vectors: List[List[float]] = []
        for text in texts:
            checksum = sum(ord(char) for char in text) % 997
            vectors.append([round((checksum + offset) / 997.0, 6) for offset in range(8)])
        return vectors


UrlOpen = Callable[..., Any]
Sleep = Callable[[float], None]


class OpenAICompatibleModelGateway(ModelGateway):
    """Small synchronous adapter for OpenAI-compatible chat and embedding APIs.

    The runtime invokes the gateway from worker threads, so a synchronous
    stdlib transport keeps the required dependency surface minimal. The
    ``opener`` and ``sleep`` hooks make retry and protocol behavior testable
    without a live provider.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_seconds: Optional[float] = None,
        app_settings: Settings = settings,
        opener: Optional[UrlOpen] = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        resolved_base_url = (
            app_settings.model_gateway_base_url if base_url is None else base_url
        )
        self.base_url = self._validate_base_url(resolved_base_url)
        self.api_key = (app_settings.model_gateway_api_key if api_key is None else api_key or "").strip()
        self.timeout_seconds = self._validate_timeout(
            app_settings.model_gateway_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        self.max_retries = self._validate_retries(
            app_settings.model_gateway_max_retries if max_retries is None else max_retries
        )
        self.backoff_seconds = self._validate_backoff(
            app_settings.model_gateway_backoff_seconds if backoff_seconds is None else backoff_seconds
        )
        self._opener = opener or urlopen
        self._sleep = sleep

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        return self.chat_with_context(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_with_context(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        context: Optional[ModelCallContext] = None,
    ) -> Dict[str, Any]:
        request_messages: List[Dict[str, str]] = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(dict(message) for message in messages)
        payload = self._request_json(
            "/chat/completions",
            {
                "model": model,
                "messages": request_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            context=context,
        )
        choices = payload.get("choices") if isinstance(payload, Mapping) else None
        if not isinstance(choices, list) or not choices:
            raise ModelGatewayResponseError("model response is missing choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ModelGatewayResponseError("model response choice is not an object")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ModelGatewayResponseError("model response choice is missing message")
        raw_content = message.get("content")
        tool_calls = message.get("tool_calls")
        # OpenAI-compatible providers may omit content when the assistant
        # emits a function/tool call.  Preserve the structured calls for the
        # OpenCode ReAct loop instead of treating that response as malformed.
        if raw_content is None and isinstance(tool_calls, list):
            content = ""
        else:
            content = self._content_text(raw_content)
        result: Dict[str, Any] = {
            "type": "tool_call" if isinstance(tool_calls, list) and tool_calls else "final",
            "content": content,
            "model": payload.get("model") or model,
            "reason_code": choice.get("finish_reason") or "remote",
        }
        if isinstance(tool_calls, list) and tool_calls:
            result["tool_calls"] = list(tool_calls)
        if payload.get("id") is not None:
            result["id"] = payload["id"]
        if isinstance(payload.get("usage"), Mapping):
            result["usage"] = dict(payload["usage"])
        return result

    def embed(self, *, model: str, texts: List[str]) -> List[List[float]]:
        return self.embed_with_context(model=model, texts=texts)

    def embed_with_context(
        self,
        *,
        model: str,
        texts: List[str],
        context: Optional[ModelCallContext] = None,
    ) -> List[List[float]]:
        if not texts:
            return []
        payload = self._request_json(
            "/embeddings",
            {"model": model, "input": list(texts)},
            context=context,
        )
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ModelGatewayResponseError("embedding response data length does not match input")
        ordered = sorted(data, key=lambda item: item.get("index", 0) if isinstance(item, Mapping) else 0)
        vectors: List[List[float]] = []
        for item in ordered:
            if not isinstance(item, Mapping) or not isinstance(item.get("embedding"), list):
                raise ModelGatewayResponseError("embedding response item is missing embedding")
            vector = item["embedding"]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
                raise ModelGatewayResponseError("embedding response contains a non-numeric value")
            vectors.append([float(value) for value in vector])
        return vectors

    def _request_json(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        context: Optional[ModelCallContext] = None,
    ) -> Dict[str, Any]:
        last_error: Optional[ModelGatewayError] = None
        budget_deadline = self._budget_deadline(context)
        for attempt in range(self.max_retries + 1):
            self._check_context(context, budget_deadline)
            try:
                result = self._perform_request(
                    path,
                    payload,
                    timeout_seconds=self._context_timeout(context, budget_deadline),
                    context=context,
                )
                self._check_context(context, budget_deadline)
                return result
            except ModelGatewayError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(exc, attempt)
                if budget_deadline is not None:
                    remaining = max(0.0, budget_deadline - time.monotonic())
                    delay = min(delay, remaining)
                if delay > 0:
                    self._sleep(delay)
        # The loop either returns or raises; this guard keeps type checkers
        # honest if the retry policy is changed later.
        assert last_error is not None
        raise last_error

    def _perform_request(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        timeout_seconds: Optional[float] = None,
        context: Optional[ModelCallContext] = None,
    ) -> Dict[str, Any]:
        self._check_context(context)
        request = Request(
            self._endpoint(path),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            response = self._opener(request, timeout=timeout_seconds or self.timeout_seconds)
        except HTTPError as exc:
            body = self._read_error_body(exc)
            raise self._http_error(exc.code, getattr(exc, "headers", None), body) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise ModelGatewayTimeout("model gateway request timed out") from exc
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise ModelGatewayTimeout("model gateway request timed out") from exc
            raise ModelGatewayTransportError("model gateway transport failed") from exc
        except OSError as exc:
            raise ModelGatewayTransportError("model gateway transport failed") from exc

        try:
            status_code = self._response_status(response)
            headers = getattr(response, "headers", None)
            body = self._read_response_body(response)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if status_code < 200 or status_code >= 300:
            raise self._http_error(status_code, headers, body)
        return self._decode_json(body)

    def _budget_deadline(self, context: Optional[ModelCallContext]) -> Optional[float]:
        if context is None:
            return None
        if context.deadline_monotonic is not None:
            return context.deadline_monotonic
        if context.timeout_seconds is not None:
            return time.monotonic() + max(0.0, float(context.timeout_seconds))
        return None

    def _context_timeout(
        self,
        context: Optional[ModelCallContext],
        budget_deadline: Optional[float],
    ) -> float:
        timeout = self.timeout_seconds
        if context is not None:
            timeout = context.remaining_seconds(timeout)
        if budget_deadline is not None:
            timeout = min(timeout, max(0.0, budget_deadline - time.monotonic()))
        if timeout <= 0:
            raise ModelGatewayTimeout("model gateway request timed out")
        return timeout

    @staticmethod
    def _check_context(
        context: Optional[ModelCallContext],
        budget_deadline: Optional[float] = None,
    ) -> None:
        if context is not None and context.is_cancelled():
            raise ModelGatewayCancelled("model gateway request cancelled")
        if budget_deadline is not None and time.monotonic() >= budget_deadline:
            raise ModelGatewayTimeout("model gateway request timed out")

    def _endpoint(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "qingling-agent-model-gateway/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _validate_base_url(value: Optional[str]) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ModelGatewayConfigurationError("model gateway base URL is required")
        candidate = value.strip().rstrip("/")
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ModelGatewayConfigurationError("model gateway base URL must be an HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ModelGatewayConfigurationError("model gateway base URL cannot contain query or fragment")
        return candidate

    @staticmethod
    def _validate_timeout(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelGatewayConfigurationError("model gateway timeout must be a number") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ModelGatewayConfigurationError("model gateway timeout must be greater than zero")
        return parsed

    @staticmethod
    def _validate_retries(value: Any) -> int:
        if isinstance(value, bool):
            raise ModelGatewayConfigurationError("model gateway max retries must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ModelGatewayConfigurationError("model gateway max retries must be an integer") from exc
        if parsed < 0 or parsed > 10:
            raise ModelGatewayConfigurationError("model gateway max retries must be between 0 and 10")
        return parsed

    @staticmethod
    def _validate_backoff(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelGatewayConfigurationError("model gateway backoff must be a number") from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ModelGatewayConfigurationError("model gateway backoff must be non-negative")
        return parsed

    @staticmethod
    def _response_status(response: Any) -> int:
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else 200
        try:
            return int(status)
        except (TypeError, ValueError) as exc:
            raise ModelGatewayResponseError("model gateway response has an invalid status") from exc

    @staticmethod
    def _read_response_body(response: Any) -> bytes:
        try:
            body = response.read()
        except (OSError, socket.timeout, TimeoutError) as exc:
            raise ModelGatewayTransportError("failed to read model gateway response") from exc
        if isinstance(body, str):
            return body.encode("utf-8")
        if not isinstance(body, (bytes, bytearray)):
            raise ModelGatewayResponseError("model gateway response body is not bytes")
        return bytes(body)

    @staticmethod
    def _read_error_body(error: HTTPError) -> bytes:
        try:
            body = error.read()
        except (OSError, socket.timeout, TimeoutError):
            return b""
        if isinstance(body, str):
            return body.encode("utf-8")
        return bytes(body) if isinstance(body, (bytes, bytearray)) else b""

    @staticmethod
    def _decode_json(body: bytes) -> Dict[str, Any]:
        if not body:
            raise ModelGatewayResponseError("model gateway returned an empty response")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelGatewayResponseError("model gateway returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ModelGatewayResponseError("model gateway response must be a JSON object")
        return value

    @classmethod
    def _http_error(cls, status_code: int, headers: Any, body: bytes) -> ModelGatewayHTTPError:
        detail = "model gateway request failed"
        if body:
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = body.decode("utf-8", errors="replace")
            if isinstance(parsed, Mapping):
                error = parsed.get("error")
                if isinstance(error, Mapping):
                    detail = str(error.get("message") or error.get("code") or detail)
                elif error:
                    detail = str(error)
                elif parsed.get("message"):
                    detail = str(parsed["message"])
            elif parsed:
                detail = str(parsed)
        retry_after_ms = cls._retry_after_ms(headers)
        return ModelGatewayHTTPError(
            status_code,
            f"HTTP {status_code}: {redact_text(detail, max_length=512)}",
            retry_after_ms=retry_after_ms,
        )

    @staticmethod
    def _retry_after_ms(headers: Any) -> Optional[int]:
        if headers is None:
            return None
        getter = getattr(headers, "get", None)
        value = getter("Retry-After") if callable(getter) else None
        if value is None:
            return None
        try:
            seconds = float(str(value).strip())
            if math.isfinite(seconds) and seconds >= 0:
                return int(seconds * 1000)
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(str(value))
            seconds = retry_at.timestamp() - time.time()
            return max(0, int(seconds * 1000))
        except (TypeError, ValueError, OverflowError):
            return None

    def _retry_delay(self, error: ModelGatewayError, attempt: int) -> float:
        if error.retry_after_ms is not None:
            return max(0.0, error.retry_after_ms / 1000.0)
        return self.backoff_seconds * (2**attempt)

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, Mapping) and item.get("text") is not None
            ]
            if parts:
                return "".join(parts)
        raise ModelGatewayResponseError("model response message is missing text content")


# Compatibility names for callers that prefer a shorter transport-oriented
# class name or an explicit provider name.
HTTPModelGateway = OpenAICompatibleModelGateway
OpenAIModelGateway = OpenAICompatibleModelGateway


def build_model_gateway(app_settings: Settings = settings) -> ModelGateway:
    """Build the model adapter selected by application configuration.

    An empty base URL intentionally selects the deterministic offline adapter,
    which keeps local development and tests independent of external services.
    Once a base URL is configured, construction is fail-fast so malformed
    provider settings cannot silently fall back to a different model.
    """

    base_url = str(getattr(app_settings, "model_gateway_base_url", "") or "").strip()
    if not base_url:
        return DeterministicModelGateway()
    return OpenAICompatibleModelGateway(app_settings=app_settings)
