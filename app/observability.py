"""Small, dependency-free helpers for request correlation and safe telemetry."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict


_CORRELATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|passwd|secret|private[_-]?key|credential|jwt|token)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|passwd|secret|private[_-]?key|credential|jwt|token)\b\s*[:=]\s*)([^\s,;}&]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_JWT_PATTERN = re.compile(r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b")
_EMAIL_PATTERN = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(\+?86[- ]?)?(1\d{2})\d{4}(\d{4})(?!\d)")

REDACTED = "***REDACTED***"


def new_correlation_id() -> str:
    """Return a log-safe identifier that is safe to place in HTTP headers."""

    return uuid.uuid4().hex


def normalize_correlation_id(value: Any) -> str:
    """Accept a bounded client correlation ID, or fail closed to a new ID."""

    if isinstance(value, str) and _CORRELATION_PATTERN.fullmatch(value):
        return value
    return new_correlation_id()


def redact_text(value: str, *, max_length: int = 1024) -> str:
    """Redact common credentials and PII without trying to parse arbitrary logs."""

    text = value
    text = _BEARER_PATTERN.sub("Bearer " + REDACTED, text)
    text = _JWT_PATTERN.sub(REDACTED, text)
    text = _SECRET_ASSIGNMENT_PATTERN.sub(lambda match: match.group(1) + REDACTED, text)
    text = _EMAIL_PATTERN.sub(lambda match: match.group(1) + REDACTED + match.group(2), text)
    text = _PHONE_PATTERN.sub(lambda match: (match.group(1) or "") + match.group(2) + "****" + match.group(3), text)
    if len(text) > max_length:
        return text[: max_length - 12] + "...<truncated>"
    return text


def redact_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Recursively redact telemetry values and cap pathological nesting."""

    if depth > 8:
        return "<max-depth>"
    if _SECRET_KEY_PATTERN.search(key):
        return REDACTED
    if isinstance(value, dict):
        return {str(item_key): redact_value(item_value, key=str(item_key), depth=depth + 1) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value))


def sanitize_audit_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return a detached, redacted audit record suitable for in-process storage."""

    return redact_value(event)
