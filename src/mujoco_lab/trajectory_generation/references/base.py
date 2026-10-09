"""Common interface for online reference generators."""

from mujoco_lab.trajectory_generation.target import ControlTarget


class ReferenceGenerator:
    """Retain reference state while goals change between fixed control ticks.

    ``compute()`` advances exactly one ``dt`` and returns an owned target snapshot.
    ``state()`` reads the current reference without advancing time. A reset clears
    runtime state and pending goals while retaining algorithm configuration.
    """

    dt: float

    def reset(self, initial: ControlTarget) -> None:
        """Restore initial position and derivatives without reconfiguring the generator."""
        raise NotImplementedError

    def set_goal(self, target: ControlTarget) -> None:
        """Replace the requested goal while preserving the current reference state."""
        raise NotImplementedError

    def state(self) -> ControlTarget:
        """Return an owned snapshot at the current reference time."""
        raise NotImplementedError

    def compute(self) -> ControlTarget:
        """Generate the next reference, advancing time by exactly one control tick."""
        raise NotImplementedError
