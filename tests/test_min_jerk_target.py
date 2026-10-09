"""Boundary continuity and analytic derivatives of minimum-jerk targets."""

import numpy as np
import pytest

from mujoco_lab.assets.robot.robot import Constraints, ReferenceConfig
from mujoco_lab.trajectory_generation import (
    ControlTarget,
    create_reference_generator,
    min_jerk,
    min_jerk_duration,
)
from mujoco_lab.trajectory_generation.references.min_jerk import MinJerkReference


@pytest.mark.parametrize("parameter", ["control_dt", "duration"])
def test_zero_reference_time_parameters_are_rejected(parameter):
    with pytest.raises(ValueError, match="must be positive"):
        MinJerkReference(
            ControlTarget(np.zeros(1)),
            Constraints(["joint"], [[-1, 1]], [1], [1]),
            **{parameter: 0.0},
        )


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


def test_constrained_distant_goal_extends_time_and_bounds_reference_derivatives():
    constraint = Constraints(["joint"], [[-1, 1]], [1.75], [1.75])
    start, end = ControlTarget(np.zeros(1)), ControlTarget(np.full(1, 0.1))
    duration = min_jerk_duration(start, end, 0.01, constraint)
    np.testing.assert_allclose(duration, np.sqrt((10 / np.sqrt(3)) * 0.1 / 1.75))
    early = min_jerk(start, end, 0.01, duration)
    assert early.position[0] < 0.01
    for time in np.linspace(0, duration, 301):
        target = min_jerk(start, end, time, duration)
        assert np.abs(target.velocity[0]) <= 1.75 + 1e-10
        assert np.abs(target.acceleration[0]) <= 1.75 + 1e-10
    np.testing.assert_allclose(target.position, end.position)


def test_constrained_moving_boundary_preserves_continuity_and_limits():
    constraint = Constraints(["joint"], [[-1, 1]], [0.5], [0.8])
    start = ControlTarget(np.zeros(1), np.full(1, 0.2), np.full(1, 0.1))
    end = ControlTarget(np.full(1, 0.3))
    duration = min_jerk_duration(start, end, 0.05, constraint)
    initial = min_jerk(start, end, 0, duration)
    for field in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(initial, field), getattr(start, field))
    for time in np.linspace(0, 3, 301):
        target = min_jerk(start, end, time, duration)
        assert np.abs(target.velocity[0]) <= 0.5 + 1e-10
        assert np.abs(target.acceleration[0]) <= 0.8 + 1e-10
    np.testing.assert_allclose(target.position, end.position)


def test_quintic_extends_short_goal_time_and_replans_continuously():
    limits = Constraints(["joint"], [[-1, 1]], [0.5], [0.8])
    reference = create_reference_generator(
        ReferenceConfig(name="min_jerk"), ControlTarget(np.zeros(1)), limits
    )
    reference.set_goal(ControlTarget(np.full(1, 0.1)))
    for _ in range(20):
        target = reference.compute()
    assert 0 < target.position[0] < 0.1
    reference.set_goal(ControlTarget(np.full(1, -0.1)))
    for field in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(reference.state(), field), getattr(target, field))
    for _ in range(1500):
        target = reference.compute()
        assert abs(target.velocity[0]) <= 0.5 + 1e-10
        assert abs(target.acceleration[0]) <= 0.8 + 1e-10
    np.testing.assert_allclose(target.position, -0.1, atol=1e-12)


def test_plain_quintic_uses_fixed_time_without_derivative_limits():
    limits = Constraints(["joint"], [[-1, 1]], [0.1], [0.8])
    reference = create_reference_generator(
        ReferenceConfig(name="min_jerk", constrain=False), ControlTarget(np.zeros(1)), limits
    )
    reference.set_goal(ControlTarget(np.full(1, 0.1)))
    midway = None
    for _ in range(5):
        target = reference.compute()
        if midway is None:
            midway = target
    assert abs(midway.velocity[0]) > limits.velocity_limit[0]
    np.testing.assert_allclose(target.position, 0.1, atol=1e-12)
