"""Hold raw joint-position goals with zero desired velocity and acceleration."""

from mujoco_lab.trajectory_generation.references.base import ReferenceGenerator
from mujoco_lab.trajectory_generation.target import ControlTarget


class PositionHoldReference(ReferenceGenerator):
    """Apply each new position immediately and retain it between policy actions."""

    def __init__(self, initial: ControlTarget, *, control_dt: float) -> None:
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        self.dt = control_dt
        self.reset(initial)

    def reset(self, initial: ControlTarget) -> None:
        self._target = ControlTarget(initial.position)

    def set_goal(self, target: ControlTarget) -> None:
        self._target = ControlTarget(target.position)

    def state(self) -> ControlTarget:
        return ControlTarget(self._target.position)

    def compute(self) -> ControlTarget:
        return self.state()
