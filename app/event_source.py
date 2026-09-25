"""Read-only security event source adapters.

The runtime treats event data as untrusted business input. This module only
retrieves normalized JSON and never exposes mutation methods.
"""

from __future__ import annotations

import json
import math
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .config import Settings, settings
from .observability import redact_text


class EventSourceError(RuntimeError):
    code = "EVENT_SOURCE_ERROR"
    category = "event_source"
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


class EventSourceConfigurationError(EventSourceError):
    code = "EVENT_SOURCE_CONFIGURATION_ERROR"
    category = "configuration"


class EventSourceTimeout(EventSourceError):
    code = "EVENT_SOURCE_TIMEOUT"
    category = "timeout"
    retryable = True


class EventSourceTransportError(EventSourceError):
    code = "EVENT_SOURCE_TRANSPORT_ERROR"
    category = "transport"
    retryable = True


class EventSourceHTTPError(EventSourceError):
    code = "EVENT_SOURCE_HTTP_ERROR"
    category = "provider"

    def __init__(self, status_code: int, message: str, *, retry_after_ms: Optional[int] = None) -> None:
        super().__init__(
            message,
            retryable=status_code in {408, 425, 429} or status_code >= 500,
            retry_after_ms=retry_after_ms,
            status_code=status_code,
        )


class EventSourceResponseError(EventSourceError):
    code = "EVENT_SOURCE_RESPONSE_ERROR"
    category = "response"


class EventSourceCancelled(EventSourceError):
    code = "EVENT_SOURCE_CANCELLED"
    category = "execution"


class EventRequestContext(Protocol):
    run_id: str
    tenant_id: str
    user_id: str
    correlation_id: str

    def remaining_seconds(self, default: float) -> float: ...
    def is_cancelled(self) -> bool: ...


class EventSource(ABC):
    @abstractmethod
    def search(self, *, query: str, context: EventRequestContext) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get(self, *, event_id: str, context: EventRequestContext) -> Dict[str, Any]:
        raise NotImplementedError


class DemoEventSource(EventSource):
    """Offline-safe source that makes simulated data explicit."""

    def search(self, *, query: str, context: EventRequestContext) -> Dict[str, Any]:
        return {
            "query": query,
            "items": [],
            "total": 0,
            "source": "demo-event-source",
            "simulated": True,
            "message": "真实事件 API 尚未接入，当前返回演示结果。",
        }

    def get(self, *, event_id: str, context: EventRequestContext) -> Dict[str, Any]:
        return {
            "event_id": event_id,
            "event": None,
            "source": "demo-event-source",
            "simulated": True,
            "message": "真实事件 API 尚未接入，当前返回演示结果。",
        }


UrlOpen = Callable[..., Any]
Sleep = Callable[[float], None]


class HTTPEventSource(EventSource):
    """Tenant-scoped adapter for a normalized read-only event HTTP API."""

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
        self.base_url = self._validate_base_url(
            app_settings.event_api_base_url if base_url is None else base_url
        )
        self.api_key = (app_settings.event_api_key if api_key is None else api_key or "").strip()
        self.timeout_seconds = self._validate_positive_float(
            app_settings.event_api_timeout_seconds if timeout_seconds is None else timeout_seconds,
            "event API timeout",
        )
        self.max_retries = self._validate_retries(
            app_settings.event_api_max_retries if max_retries is None else max_retries
        )
        self.backoff_seconds = self._validate_non_negative_float(
            app_settings.event_api_backoff_seconds if backoff_seconds is None else backoff_seconds,
            "event API backoff",
        )
        self._opener = opener or urlopen
        self._sleep = sleep

    def search(self, *, query: str, context: EventRequestContext) -> Dict[str, Any]:
        payload = self._request_json(
            "POST",
            "/events/search",
            {"query": query},
            context=context,
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise EventSourceResponseError("event search response is missing an items array")
        total = payload.get("total", len(items))
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise EventSourceResponseError("event search response has an invalid total")
        return {
            "query": query,
            "items": items,
            "total": total,
            "source": str(payload.get("source") or "http-event-source"),
            "simulated": False,
        }

    def get(self, *, event_id: str, context: EventRequestContext) -> Dict[str, Any]:
        payload = self._request_json(
            "GET",
            "/events/" + quote(event_id, safe=""),
            None,
            context=context,
        )
        event = payload.get("event")
        if event is not None and not isinstance(event, Mapping):
            raise EventSourceResponseError("event detail response has an invalid event object")
        return {
            "event_id": event_id,
            "event": dict(event) if isinstance(event, Mapping) else None,
            "source": str(payload.get("source") or "http-event-source"),
            "simulated": False,
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
        *,
        context: EventRequestContext,
    ) -> Dict[str, Any]:
        last_error: Optional[EventSourceError] = None
        for attempt in range(self.max_retries + 1):
            self._check_context(context)
            try:
                timeout = context.remaining_seconds(self.timeout_seconds)
                if timeout <= 0:
                    raise EventSourceTimeout("event API request timed out")
                result = self._perform_request(method, path, payload, context=context, timeout=timeout)
                self._check_context(context)
                return result
            except EventSourceError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(exc, attempt)
                remaining = context.remaining_seconds(self.timeout_seconds)
                if remaining <= 0:
                    raise EventSourceTimeout("event API request timed out") from exc
                delay = min(delay, remaining)
                if delay > 0:
                    cancel_event = getattr(context, "cancel_event", None)
                    if cancel_event is not None and cancel_event.wait(delay):
                        raise EventSourceCancelled("event API request cancelled") from exc
                    if cancel_event is None:
                        self._sleep(delay)
        assert last_error is not None
        raise last_error

    def _perform_request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
        *,
        context: EventRequestContext,
        timeout: float,
    ) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.base_url.rstrip("/") + "/" + path.lstrip("/"),
            data=data,
            headers=self._headers(context),
            method=method,
        )
        try:
            response = self._opener(request, timeout=timeout)
        except HTTPError as exc:
            body = self._read_error_body(exc)
            raise self._http_error(exc.code, getattr(exc, "headers", None), body) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise EventSourceTimeout("event API request timed out") from exc
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise EventSourceTimeout("event API request timed out") from exc
            raise EventSourceTransportError("event API transport failed") from exc
        except OSError as exc:
            raise EventSourceTransportError("event API transport failed") from exc

        try:
            status = self._response_status(response)
            headers = getattr(response, "headers", None)
            body = self._read_response_body(response)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if status < 200 or status >= 300:
            raise self._http_error(status, headers, body)
        return self._decode_json(body)

    def _headers(self, context: EventRequestContext) -> Dict[str, str]:
        header_values = {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "correlation_id": context.correlation_id,
            "run_id": context.run_id,
        }
        for name, value in header_values.items():
            if not isinstance(value, str) or not value or len(value) > 256 or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
                raise EventSourceConfigurationError(f"event API context field {name} is invalid")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "qingling-agent-event-source/0.1",
            "X-Tenant-ID": context.tenant_id,
            "X-User-ID": context.user_id,
            "X-Correlation-ID": context.correlation_id,
            "X-Run-ID": context.run_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _check_context(context: EventRequestContext) -> None:
        if context.is_cancelled():
            raise EventSourceCancelled("event API request cancelled")

    @staticmethod
    def _validate_base_url(value: Optional[str]) -> str:
        if not isinstance(value, str) or not value.strip():
            raise EventSourceConfigurationError("event API base URL is required")
        candidate = value.strip().rstrip("/")
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EventSourceConfigurationError("event API base URL must be an HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise EventSourceConfigurationError("event API base URL cannot contain query or fragment")
        return candidate

    @staticmethod
    def _validate_positive_float(value: Any, label: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise EventSourceConfigurationError(f"{label} must be a number") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise EventSourceConfigurationError(f"{label} must be greater than zero")
        return parsed

    @staticmethod
    def _validate_non_negative_float(value: Any, label: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise EventSourceConfigurationError(f"{label} must be a number") from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise EventSourceConfigurationError(f"{label} must be non-negative")
        return parsed

    @staticmethod
    def _validate_retries(value: Any) -> int:
        if isinstance(value, bool):
            raise EventSourceConfigurationError("event API max retries must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise EventSourceConfigurationError("event API max retries must be an integer") from exc
        if parsed < 0 or parsed > 10:
            raise EventSourceConfigurationError("event API max retries must be between 0 and 10")
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
            raise EventSourceResponseError("event API response has an invalid status") from exc

    @staticmethod
    def _read_response_body(response: Any) -> bytes:
        try:
            body = response.read()
        except (OSError, socket.timeout, TimeoutError) as exc:
            raise EventSourceTransportError("failed to read event API response") from exc
        if isinstance(body, str):
            return body.encode("utf-8")
        if not isinstance(body, (bytes, bytearray)):
            raise EventSourceResponseError("event API response body is not bytes")
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
            raise EventSourceResponseError("event API returned an empty response")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EventSourceResponseError("event API returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise EventSourceResponseError("event API response must be a JSON object")
        return value

    @classmethod
    def _http_error(cls, status: int, headers: Any, body: bytes) -> EventSourceHTTPError:
        detail = "event API request failed"
        if body:
            try:
                value = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = body.decode("utf-8", errors="replace")
            if isinstance(value, Mapping):
                error = value.get("error")
                if isinstance(error, Mapping):
                    detail = str(error.get("message") or error.get("code") or detail)
                elif value.get("message"):
                    detail = str(value["message"])
            elif value:
                detail = str(value)
        return EventSourceHTTPError(
            status,
            f"HTTP {status}: {redact_text(detail, max_length=512)}",
            retry_after_ms=cls._retry_after_ms(headers),
        )

    @staticmethod
    def _retry_after_ms(headers: Any) -> Optional[int]:
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
            return max(0, int((retry_at.timestamp() - time.time()) * 1000))
        except (TypeError, ValueError, OverflowError):
            return None

    def _retry_delay(self, error: EventSourceError, attempt: int) -> float:
        if error.retry_after_ms is not None:
            return max(0.0, error.retry_after_ms / 1000.0)
        return self.backoff_seconds * (2**attempt)


def build_event_source(app_settings: Settings = settings) -> EventSource:
    base_url = str(getattr(app_settings, "event_api_base_url", "") or "").strip()
    if not base_url:
        return DemoEventSource()
    return HTTPEventSource(app_settings=app_settings)
