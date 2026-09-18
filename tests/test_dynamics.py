import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab import create_robot
from mujoco_lab.state.dynamics import Dynamics


def test_jacobian_rows_match_world_position_and_rotation_derivatives():
    model, data = create_robot("forte", environment="warehouse")
    dynamics = Dynamics(model, data)
    jacobian = dynamics.get_jacobian("ee_site").copy()
    assert jacobian.shape == (6, model.nv)
    np.testing.assert_array_equal(dynamics.get_frame_position("ee_site"), data.site("ee_site").xpos)

    plus = mujoco.MjData(model)
    minus = mujoco.MjData(model)
    epsilon = 1e-6
    finite_difference = np.empty_like(jacobian)
    for column in range(model.nv):
        velocity = np.zeros(model.nv)
        velocity[column] = 1.0
        plus.qpos[:] = data.qpos
        minus.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, plus.qpos, velocity, epsilon)
        mujoco.mj_integratePos(model, minus.qpos, velocity, -epsilon)
        mujoco.mj_forward(model, plus)
        mujoco.mj_forward(model, minus)
        forward = plus.site("ee_site")
        backward = minus.site("ee_site")
        finite_difference[:3, column] = (forward.xpos - backward.xpos) / (2 * epsilon)
        rotation_delta = forward.xmat.reshape(3, 3) @ backward.xmat.reshape(3, 3).T
        finite_difference[3:, column] = Rotation.from_matrix(rotation_delta).as_rotvec() / (
            2 * epsilon
        )
    np.testing.assert_allclose(jacobian, finite_difference, rtol=1e-6, atol=1e-8)


def test_getters_reuse_independent_buffers_and_allow_persistent_copies():
    model, data = create_robot("forte")
    dynamics = Dynamics(model, data)
    jacobian = dynamics.get_jacobian("ee_site")
    jacobian_snapshot = jacobian.copy()
    mass = dynamics.get_mass_matrix()
    mass_snapshot = mass.copy()
    assert mass.shape == (model.nv, model.nv)
    assert not np.shares_memory(jacobian, mass)
    np.testing.assert_array_equal(jacobian, jacobian_snapshot)

    mount_jacobian = dynamics.get_jacobian("robot_mount")
    assert np.shares_memory(jacobian, mount_jacobian)
    np.testing.assert_array_equal(mount_jacobian, np.zeros((6, model.nv)))
    assert np.any(jacobian_snapshot != 0)
    np.testing.assert_array_equal(mass, mass_snapshot)

    data.qpos[3] += 0.25
    mujoco.mj_forward(model, data)
    updated_mass = dynamics.get_mass_matrix()
    assert np.shares_memory(mass, updated_mass)
    assert not np.allclose(updated_mass, mass_snapshot)
    assert data.time == 0.0
