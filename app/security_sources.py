"""Read-only source contracts and in-memory implementations for security data."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from threading import RLock
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Protocol, TypeVar

from .security_models import AssetRecord, ExposureRecord, VulnerabilityRecord


T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SourceContext:
    """Trusted request context passed explicitly to every source query."""

    tenant_id: str
    user_id: str = "anonymous"
    run_id: str = ""
    correlation_id: str = ""
    roles: tuple[str, ...] = ()
    deadline_monotonic: Optional[float] = None
    cancel_event: Any = None

    def check_active(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TimeoutError("security source query cancelled")
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise TimeoutError("security source deadline exceeded")


@dataclass(frozen=True)
class SourceResult(Generic[T]):
    items: List[T]
    source_id: str
    fetched_at: datetime = field(default_factory=utc_now)
    partial: bool = False
    errors: List[str] = field(default_factory=list)


class AssetSource(Protocol):
    source_id: str

    def search(self, query: Mapping[str, Any], context: SourceContext) -> SourceResult[AssetRecord]: ...


class VulnerabilitySource(Protocol):
    source_id: str

    def search(self, query: Mapping[str, Any], context: SourceContext) -> SourceResult[VulnerabilityRecord]: ...


class ExposureSource(Protocol):
    source_id: str

    def search(self, query: Mapping[str, Any], context: SourceContext) -> SourceResult[ExposureRecord]: ...


class _MemorySource(Generic[T]):
    """Simple test/development source. Query results are always tenant scoped."""

    record_type: Any

    def __init__(self, source_id: str, records: Iterable[T] = ()) -> None:
        self.source_id = source_id
        self._records: List[T] = list(records)
        self._lock = RLock()

    def add(self, record: T) -> None:
        if not isinstance(record, self.record_type):
            raise TypeError(f"record must be {self.record_type.__name__}")
        with self._lock:
            self._records.append(record)

    def replace(self, records: Iterable[T]) -> None:
        normalized = list(records)
        if any(not isinstance(record, self.record_type) for record in normalized):
            raise TypeError(f"records must be {self.record_type.__name__}")
        with self._lock:
            self._records = normalized

    def search(self, query: Mapping[str, Any], context: SourceContext) -> SourceResult[T]:
        context.check_active()
        query = dict(query or {})
        target_keys = ("asset_id", "ip", "hostname")
        selectors = {key: str(query[key]).strip() for key in target_keys if query.get(key) not in (None, "")}
        with self._lock:
            records = list(self._records)
        selected: List[T] = []
        for record in records:
            if getattr(record, "tenant_id", None) != context.tenant_id:
                continue
            if selectors:
                matched = any(str(getattr(record, key, "") or "") == value for key, value in selectors.items())
                if not matched:
                    continue
            selected.append(record)
        context.check_active()
        return SourceResult(items=selected, source_id=self.source_id)


class InMemoryAssetSource(_MemorySource[AssetRecord]):
    record_type = AssetRecord

    def __init__(self, records: Iterable[AssetRecord] = (), source_id: str = "asset.inventory.memory") -> None:
        super().__init__(source_id, records)


class InMemoryVulnerabilitySource(_MemorySource[VulnerabilityRecord]):
    record_type = VulnerabilityRecord

    def __init__(self, records: Iterable[VulnerabilityRecord] = (), source_id: str = "vulnerability.inventory.memory") -> None:
        super().__init__(source_id, records)


class InMemoryExposureSource(_MemorySource[ExposureRecord]):
    record_type = ExposureRecord

    def __init__(self, records: Iterable[ExposureRecord] = (), source_id: str = "exposure.inventory.memory") -> None:
        super().__init__(source_id, records)
