import pytest

from mujoco_lab.utils import StateMachine


def test_state_machine_accepts_only_configured_transitions():
    machine = StateMachine("idle", {"idle": {"running"}, "running": {"idle"}})

    machine.transition("running")
    assert machine.get_state() == "running"
    machine.transition("idle")
    assert machine.get_state() == "idle"

    with pytest.raises(RuntimeError, match="Invalid state transition"):
        machine.transition("idle")
    assert machine.get_state() == "idle"
