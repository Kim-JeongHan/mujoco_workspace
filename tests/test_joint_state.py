"""Tests for owned JointState projections."""

import numpy as np

from mujoco_lab.state.joint_state import JointState


def test_select_reorders_scalar_joints_and_returns_owned_arrays():
    state = JointState(
        time=1.5,
        qpos=np.array([1, 2, 3], dtype=np.float32),
        qvel=np.array([10, 20, 30], dtype=np.float64),
        bias_forces=np.array([100, 200, 300], dtype=np.float64),
    )

    selected = state.select([2, 0])

    assert selected.time == state.time
    np.testing.assert_array_equal(selected.qpos, [3, 1])
    np.testing.assert_array_equal(selected.qvel, [30, 10])
    np.testing.assert_array_equal(selected.bias_forces, [300, 100])
    assert selected.qpos.dtype == state.qpos.dtype
    assert selected.qvel.dtype == state.qvel.dtype
    assert selected.bias_forces.dtype == state.bias_forces.dtype
    assert not np.shares_memory(selected.qpos, state.qpos)
    assert not np.shares_memory(selected.qvel, state.qvel)
    assert not np.shares_memory(selected.bias_forces, state.bias_forces)

    selected.qpos[0] = -1
    state.qvel[2] = -10
    assert state.qpos[2] == 3
    assert selected.qvel[0] == 30
