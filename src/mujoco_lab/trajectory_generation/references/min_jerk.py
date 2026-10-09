"""Online quintic replanning through the common reference interface."""

from mujoco_lab.assets.robot.robot import Constraints
from mujoco_lab.trajectory_generation.interpolation import min_jerk, min_jerk_duration
from mujoco_lab.trajectory_generation.references.base import ReferenceGenerator
from mujoco_lab.trajectory_generation.target import ControlTarget


class MinJerkReference(ReferenceGenerator):
    """Replan bounded quintics from the current reference when a goal arrives.

    ``duration`` is the requested minimum segment time; velocity and acceleration
    limits can extend it. Both initial and goal derivatives are respected. Position,
    jerk, torque, and actual robot motion are not bounded by this generator.
    With ``constrain=False``, segment duration stays fixed and reference derivatives
    are not limited; this reproduces plain quintic interpolation for comparisons.
    """

    def __init__(
        self,
        initial: ControlTarget,
        constraints: Constraints,
        *,
        control_dt: float = 0.002,
        duration: float = 0.01,
        constrain: bool = True,
    ) -> None:
        self.dt = float(control_dt)
        self._constraints = constraints
        self._minimum_duration = float(duration)
        if self.dt <= 0 or self._minimum_duration <= 0:
            raise ValueError("control_dt and duration must be positive")
        self._constrain = constrain
        self.reset(initial)

    def reset(self, initial: ControlTarget) -> None:
        """Hold the initial position while retaining its starting derivatives."""
        self._start = ControlTarget(initial.position, initial.velocity, initial.acceleration)
        self._goal = ControlTarget(initial.position)
        self._duration = self._plan_duration(self._start, self._goal)
        self._elapsed = 0.0

    def set_goal(self, target: ControlTarget) -> None:
        """Plan once from the current reference without resetting its derivatives."""
        start = self.state()
        goal = ControlTarget(target.position, target.velocity, target.acceleration)
        duration = self._plan_duration(start, goal)
        self._start, self._goal = start, goal
        self._duration = duration
        self._elapsed = 0.0

    def state(self) -> ControlTarget:
        return min_jerk(self._start, self._goal, self._elapsed, self._duration)

    def _plan_duration(self, start: ControlTarget, goal: ControlTarget) -> float:
        if not self._constrain:
            return self._minimum_duration
        return min_jerk_duration(start, goal, self._minimum_duration, self._constraints)

    def compute(self) -> ControlTarget:
        self._elapsed += self.dt
        return self.state()
