"""Shared deterministic inputs for regression tests."""
from itertools import count
from types import SimpleNamespace
from uuid import UUID

import pytest


@pytest.fixture
def phone_like_uuids(monkeypatch):
    """Valid, unique UUID4 values whose suffix resembles a telephone number.

    Patch only the store's UUID provider, not the global uuid module. This
    forces the rare random-data case without loops, sleeps or reused IDs.
    """
    sequence = count(1)
    monkeypatch.setattr("app.store.uuid", SimpleNamespace(
        uuid4=lambda: UUID(f"{next(sequence):08x}cdef4abc8defa13812345678"),
    ))
