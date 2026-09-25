import pytest

from app.state import InvalidTransition, transition


def test_run_state_machine_allows_approval_path():
    assert transition("queued", "running") == "running"
    assert transition("running", "waiting_approval") == "waiting_approval"
    assert transition("waiting_approval", "running") == "running"
    assert transition("running", "succeeded") == "succeeded"


def test_run_state_machine_rejects_terminal_restart():
    with pytest.raises(InvalidTransition):
        transition("succeeded", "running")
