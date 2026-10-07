"""Shape-preserving joint curves respect bounds without stopping at every knot."""

import numpy as np
import pytest

from mujoco_lab.control.trajectory import JointTrajectory


@pytest.mark.parametrize("smooth", [True, False])
@pytest.mark.parametrize("count", [1, 3])
def test_stationary_paths_hold_position_with_zero_duration(count, smooth):
    path = np.tile([0.2, -0.1], (count, 1))
    trajectory = JointTrajectory(path, 1.0, 1.0, smooth=smooth)
    assert trajectory.duration == 0
    for elapsed in (-1.0, 0.0, 1.0):
        target = trajectory.sample(elapsed)
        np.testing.assert_array_equal(target.position, path[0])
        np.testing.assert_array_equal(target.velocity, 0)
        np.testing.assert_array_equal(target.acceleration, 0)


@pytest.mark.parametrize("smooth", [True, False])
def test_ratio_scales_time_and_derivatives_without_changing_the_curve(smooth):
    path = np.array([[0.0, 0.0], [0.2, 0.1], [0.3, 0.4]])
    baseline = JointTrajectory(path, np.array([1.0, 2.0]), np.array([3.0, 4.0]), smooth=smooth)
    reduced = JointTrajectory(
        path, np.array([1.0, 2.0]), np.array([3.0, 4.0]), ratio=(0.5, 0.25), smooth=smooth
    )
    assert reduced.duration == pytest.approx(2 * baseline.duration)
    for elapsed in np.linspace(0, baseline.duration, 101):
        original = baseline.sample(elapsed)
        scaled = reduced.sample(2 * elapsed)
        assert original.velocity is not None and original.acceleration is not None
        assert scaled.velocity is not None and scaled.acceleration is not None
        np.testing.assert_allclose(scaled.position, original.position, atol=1e-12)
        np.testing.assert_allclose(scaled.velocity, original.velocity / 2, atol=1e-12)
        np.testing.assert_allclose(scaled.acceleration, original.acceleration / 4, atol=1e-12)
    velocity = reduced.sample(reduced.waypoint_times[1]).velocity
    assert velocity is not None
    if smooth:
        assert np.linalg.norm(velocity) > 0
    else:
        np.testing.assert_allclose(velocity, 0)


@pytest.mark.parametrize("smooth", [True, False])
def test_per_joint_ratios_bound_motion_and_preserve_input_arrays(smooth):
    path = np.array([[0.0, 0.0], [0.2, 0.1], [0.3, 0.4]])
    velocity = np.array([1.0, 2.0])
    acceleration = np.array([3.0, 4.0])
    ratios = np.array([[0.9, 1.0], [0.2, 0.3]])
    trajectory = JointTrajectory(path, velocity, acceleration, ratio=ratios, smooth=smooth)
    ratios[:] = 1  # The trajectory owns its ratio snapshot.
    for elapsed in np.linspace(0, trajectory.duration, 501):
        sample = trajectory.sample(elapsed)
        assert sample.velocity is not None and sample.acceleration is not None
        assert np.all(np.abs(sample.velocity) <= [0.9 + 1e-12, 0.4 + 1e-12])
        assert np.all(np.abs(sample.acceleration) <= [3.0 + 1e-12, 1.2 + 1e-12])
    np.testing.assert_array_equal(velocity, [1.0, 2.0])
    np.testing.assert_array_equal(acceleration, [3.0, 4.0])
    np.testing.assert_array_equal(trajectory.ratio, [[0.9, 1.0], [0.2, 0.3]])
    assert not trajectory.ratio.flags.writeable
    assert not trajectory.max_velocity.flags.writeable
    assert not trajectory.max_acceleration.flags.writeable


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize("index", [0, 1])
def test_invalid_ratio_fractions_are_rejected(value, index):
    ratios = np.ones((2, 2))
    ratios[0, index] = value
    with pytest.raises(ValueError, match="finite and in"):
        JointTrajectory(np.zeros((2, 2)), 1.0, 1.0, ratio=ratios)


@pytest.mark.parametrize("ratio", [np.array(0.5), np.ones(3), np.ones((3, 2)), np.ones((2, 1))])
def test_invalid_ratio_shapes_are_rejected(ratio):
    with pytest.raises(ValueError, match="pair"):
        JointTrajectory(np.zeros((2, 2)), 1.0, 1.0, ratio=ratio)


def test_shape_preserving_curve_keeps_waypoint_bounds_and_motion_limits():
    path = np.array([[0, 0], [0.09, 0.4], [0.1, 0.7], [0.1, 1.2], [0.07, 1.5]])
    velocity = np.array([1.0, 1.5])
    acceleration = np.array([2.0, 3.0])
    trajectory = JointTrajectory(path, velocity, acceleration, shape_preserving=True)
    for index in range(len(path) - 1):
        lower = np.minimum(path[index], path[index + 1])
        upper = np.maximum(path[index], path[index + 1])
        start, end = trajectory.waypoint_times[index : index + 2]
        for time in np.linspace(start, end, 101):
            target = trajectory.sample(time)
            assert target.velocity is not None and target.acceleration is not None
            assert np.all(target.position >= lower - 1e-12)
            assert np.all(target.position <= upper + 1e-12)
            assert np.all(np.abs(target.velocity) <= velocity + 1e-12)
            assert np.all(np.abs(target.acceleration) <= acceleration + 1e-12)
    for time in trajectory.waypoint_times[1:-1]:
        velocity_at_knot = trajectory.sample(time).velocity
        assert velocity_at_knot is not None
        assert np.linalg.norm(velocity_at_knot) > 0
    np.testing.assert_array_equal(trajectory.sample(0).velocity, 0)
    np.testing.assert_array_equal(trajectory.sample(trajectory.duration).velocity, 0)
