"""Convert action frequencies and simulated durations to discrete steps."""

import math


def max_steps_for_seconds(max_seconds: float, action_dt: float) -> int:
    """Round a simulated time limit up to cover a whole number of actions."""
    if not (0 < max_seconds < math.inf):
        raise ValueError("max_seconds must be finite and positive")
    if not (0 < action_dt < math.inf):
        raise ValueError("action_dt must be finite and positive")
    return math.ceil(max_seconds / action_dt)


def physics_steps_per_action(simulation_hz: float, action_execution_hz: float) -> int:
    """Return the action repeat without silently rounding incompatible frequencies."""
    if not (0 < simulation_hz < math.inf and 0 < action_execution_hz < math.inf):
        raise ValueError("simulation_hz and action_execution_hz must be finite and positive")
    repeat = simulation_hz / action_execution_hz
    if repeat < 1 or not math.isclose(repeat, round(repeat)):
        raise ValueError("simulation_hz must be an integer multiple of action_execution_hz")
    return round(repeat)
