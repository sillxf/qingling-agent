from __future__ import annotations

from typing import Dict, Set


class InvalidTransition(ValueError):
    pass


TERMINAL_STATES: Set[str] = {"succeeded", "failed", "cancelled", "timed_out"}

ALLOWED_TRANSITIONS: Dict[str, Set[str]] = {
    "queued": {"running", "cancelled", "timed_out"},
    "running": {
        "waiting_approval",
        "retrying",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    },
    "retrying": {"running", "failed", "cancelled", "timed_out"},
    "waiting_approval": {"running", "failed", "cancelled", "timed_out"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
    "timed_out": set(),
}


def transition(current: str, target: str) -> str:
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        raise InvalidTransition(f"cannot transition run from {current!r} to {target!r}")
    return target
