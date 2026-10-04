"""Boundary continuity and analytic derivatives of minimum-jerk targets."""

import numpy as np
import pytest

from mujoco_lab.control import ControlTarget, min_jerk_target
from mujoco_lab.control.trajectory import min_jerk


def test_stationary_boundaries_match_original_blend():
    start = ControlTarget([0.2, -0.3])
    end = ControlTarget([0.6, 0.1])
    for u in np.linspace(0, 1, 11):
        target = min_jerk_target(start, end, u * 0.4, 0.4)
        blend, slope = min_jerk(u)
        np.testing.assert_allclose(
            target.position, start.position + (end.position - start.position) * blend
        )
        np.testing.assert_allclose(
            target.velocity, (end.position - start.position) * slope / 0.4, atol=1e-13
        )


def test_replanning_preserves_nonzero_velocity_and_acceleration():
    start = ControlTarget([0.2, -0.3], [0.4, -0.2], [0.1, 0.5])
    end = ControlTarget([0.6, 0.1], [-0.2, 0.1], [0.3, -0.1])
    for elapsed, boundary in [(0, start), (0.4, end)]:
        target = min_jerk_target(start, end, elapsed, 0.4)
        for name in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(getattr(target, name), getattr(boundary, name), atol=1e-12)
    previous = min_jerk_target(start, end, 0.15, 0.4)
    assert np.any(abs(previous.velocity) > 0.1)
    following = min_jerk_target(previous, ControlTarget([0.1, -0.4]), 0, 0.2)
    for name in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(following, name), getattr(previous, name))


def test_returned_derivatives_match_position_and_velocity_changes():
    start = ControlTarget([0.2], [0.4], [0.1])
    end = ControlTarget([0.6], [-0.2], [0.3])
    time, delta = 0.15, 1e-6
    before = min_jerk_target(start, end, time - delta, 0.4)
    current = min_jerk_target(start, end, time, 0.4)
    after = min_jerk_target(start, end, time + delta, 0.4)
    np.testing.assert_allclose(
        (after.position - before.position) / (2 * delta), current.velocity, rtol=1e-8
    )
    np.testing.assert_allclose(
        (after.velocity - before.velocity) / (2 * delta), current.acceleration, rtol=1e-8
    )


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        min_jerk_target(ControlTarget([0]), ControlTarget([1]), 0, duration)
