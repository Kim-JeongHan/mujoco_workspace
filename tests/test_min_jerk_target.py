"""Boundary continuity and analytic derivatives of minimum-jerk targets."""

import numpy as np

from mujoco_lab.control import ControlTarget, min_jerk


def test_replanning_preserves_nonzero_velocity_and_acceleration():
    start = ControlTarget([0.2, -0.3], [0.4, -0.2], [0.1, 0.5])
    end = ControlTarget([0.6, 0.1], [-0.2, 0.1], [0.3, -0.1])
    for elapsed, boundary in [(0, start), (0.4, end)]:
        target = min_jerk(start, end, elapsed, 0.4)
        for name in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(getattr(target, name), getattr(boundary, name), atol=1e-12)
    previous = min_jerk(start, end, 0.15, 0.4)
    assert np.any(abs(previous.velocity) > 0.1)
    following = min_jerk(previous, ControlTarget([0.1, -0.4]), 0, 0.2)
    for name in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(following, name), getattr(previous, name))


def test_returned_derivatives_match_position_and_velocity_changes():
    start = ControlTarget([0.2], [0.4], [0.1])
    end = ControlTarget([0.6], [-0.2], [0.3])
    time, delta = 0.15, 1e-6
    before = min_jerk(start, end, time - delta, 0.4)
    current = min_jerk(start, end, time, 0.4)
    after = min_jerk(start, end, time + delta, 0.4)
    np.testing.assert_allclose(
        (after.position - before.position) / (2 * delta), current.velocity, rtol=1e-8
    )
    np.testing.assert_allclose(
        (after.velocity - before.velocity) / (2 * delta), current.acceleration, rtol=1e-8
    )
