"""Check the finite-horizon objective, continuous bounds, and owned reference state."""

import numpy as np
import pytest

from mujoco_lab.assets.robot.robot import Constraints
from mujoco_lab.trajectory_generation.references.analytic_jerk import AnalyticJerkReference
from mujoco_lab.trajectory_generation.target import ControlTarget


@pytest.mark.parametrize("parameter", ["control_dt", "response_time", "horizon"])
def test_zero_time_parameters_are_rejected(parameter):
    with pytest.raises(ValueError, match="must be positive"):
        AnalyticJerkReference(
            ControlTarget(np.zeros(1)),
            Constraints(["joint"], [[-1, 1]], [1], [1]),
            **{parameter: 0.0},
        )


@pytest.mark.parametrize("terminal_weight", [0.0, 2.0])
def test_first_jerk_matches_dense_finite_horizon_solution(terminal_weight):
    dt, tau, steps = 0.02, 0.1, 8
    state = np.array([0.2, 0.3, -0.1])
    goal = np.full(1, 0.4)
    reference = AnalyticJerkReference(
        ControlTarget(state[:1], state[1:2], state[2:3]),
        Constraints(["joint"], [[-100, 100]], [100], [100]),
        control_dt=dt,
        response_time=tau,
        horizon=steps * dt,
        terminal_weight=terminal_weight,
    )
    reference.set_goal(ControlTarget(goal))
    matrix = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
    input_vector = np.array([dt**3 / 6, dt**2 / 2, dt])
    initial = state.copy()
    initial[0] -= goal[0]
    response = np.zeros((steps, steps))
    free = np.zeros(steps)
    for row in range(steps):
        free[row] = (np.linalg.matrix_power(matrix, row + 1) @ initial)[0]
        for column in range(row + 1):
            response[row, column] = (np.linalg.matrix_power(matrix, row - column) @ input_vector)[0]
    weights = np.ones(steps)
    weights[-1] += terminal_weight
    expected = np.linalg.solve(
        response.T @ (weights[:, None] * response) + tau**6 * np.eye(steps),
        -(response.T @ (weights * free)),
    )
    reference.compute()
    np.testing.assert_allclose(reference.last_jerk, expected[0], rtol=1e-11)


def test_changing_goal_preserves_state_and_constant_jerk_integration():
    reference = AnalyticJerkReference(
        ControlTarget(np.arange(2) * 0.2),
        Constraints(["first", "second"], [[-1, 1], [-1, 1]], [1, 1], [2, 2]),
    )
    reference.set_goal(ControlTarget(np.linspace(0.1, -0.2, 2)))
    for _ in range(20):
        reference.compute()
    before = reference.state()
    goal = ControlTarget(np.linspace(-0.1, 0.3, 2))
    reference.set_goal(goal)
    goal.position[:] = 100
    for field in ("position", "velocity", "acceleration"):
        np.testing.assert_array_equal(getattr(reference.state(), field), getattr(before, field))
    after = reference.compute()
    h, jerk = reference.dt, reference.last_jerk
    np.testing.assert_allclose(
        after.position,
        before.position + h * before.velocity + h**2 * before.acceleration / 2 + h**3 * jerk / 6,
    )
    np.testing.assert_allclose(
        after.velocity, before.velocity + h * before.acceleration + h**2 * jerk / 2
    )
    np.testing.assert_allclose(after.acceleration, before.acceleration + h * jerk)
    after.position[:] = 100
    assert np.all(reference.state().position < 1)


def test_goal_reversals_obey_bounds_at_endpoints_and_interior_velocity_extrema():
    reference = AnalyticJerkReference(
        ControlTarget(np.zeros(7)),
        Constraints(
            [f"joint_{i}" for i in range(7)],
            [[-3, 3] for _ in range(7)],
            np.linspace(0.1, 0.4, 7).tolist(),
            np.linspace(0.4, 1.75, 7).tolist(),
        ),
    )
    for step in range(3000):
        if step % 250 == 0:
            reference.set_goal(ControlTarget(np.full(7, 2 if (step // 250) % 2 == 0 else -2)))
        before = reference.state()
        after = reference.compute()
        jerk = reference.last_jerk
        extremum = np.divide(-before.acceleration, jerk, out=np.zeros_like(jerk), where=jerk != 0)
        extremum = np.clip(extremum, 0, reference.dt)
        interior_velocity = (
            before.velocity + before.acceleration * extremum + jerk * extremum**2 / 2
        )
        assert np.all(np.abs(interior_velocity) <= reference.vmax + 1e-10)
        assert np.all(np.abs(after.velocity) <= reference.vmax + 1e-10)
        assert np.all(np.abs(after.acceleration) <= reference.amax + 1e-10)


def test_unrecoverable_outward_acceleration_at_speed_bound_is_reported():
    reference = AnalyticJerkReference(
        ControlTarget(np.zeros(1), np.full(1, 0.1), np.full(1, 0.2)),
        Constraints(["joint"], [[-1, 1]], [0.1], [1]),
    )
    with pytest.raises(ValueError, match="outward acceleration"):
        reference.compute()
