import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.state import JointState, RobotState


def test_jacobian_rows_match_world_position_and_rotation_derivatives():
    sim = Simulator(create_environment("warehouse"), robots=[RobotSpec("forte", "forte")])
    model, data = sim.model, sim.data
    state = sim.robots["forte"].state
    jacobian = state.get_jacobian("ee_site").copy()
    assert jacobian.shape == (6, model.nv)
    site = state.site_id("ee_site")
    np.testing.assert_array_equal(state.get_frame_position("ee_site"), data.site(site).xpos)

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
        forward = plus.site(site)
        backward = minus.site(site)
        finite_difference[:3, column] = (forward.xpos - backward.xpos) / (2 * epsilon)
        rotation_delta = forward.xmat.reshape(3, 3) @ backward.xmat.reshape(3, 3).T
        finite_difference[3:, column] = Rotation.from_matrix(rotation_delta).as_rotvec() / (
            2 * epsilon
        )
    np.testing.assert_allclose(jacobian, finite_difference, rtol=1e-6, atol=1e-8)


def test_getters_reuse_independent_buffers_and_allow_persistent_copies():
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("forte", "forte")])
    model, data = sim.model, sim.data
    state = sim.robots["forte"].state
    jacobian = state.get_jacobian("ee_site")
    jacobian_snapshot = jacobian.copy()
    mass = state.get_mass_matrix()
    mass_snapshot = mass.copy()
    assert mass.shape == (model.nv, model.nv)
    assert not np.shares_memory(jacobian, mass)
    np.testing.assert_array_equal(jacobian, jacobian_snapshot)

    data.qpos[0] += 0.2
    mujoco.mj_forward(model, data)
    updated_jacobian = state.get_jacobian("ee_site")
    assert np.shares_memory(jacobian, updated_jacobian)
    assert not np.allclose(updated_jacobian, jacobian_snapshot)
    np.testing.assert_array_equal(mass, mass_snapshot)

    data.qpos[3] += 0.25
    mujoco.mj_forward(model, data)
    updated_mass = state.get_mass_matrix()
    assert np.shares_memory(mass, updated_mass)
    assert not np.allclose(updated_mass, mass_snapshot)
    assert data.time == 0.0


def test_state_reads_never_advance_or_refresh_shared_physics(monkeypatch):
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("forte", "forte")])
    state = sim.robots["forte"].state
    assert isinstance(state, RobotState)
    previous = state.snapshot()
    position = state.get_frame_position("ee_site").copy()
    sim.data.qpos[0] += 0.2

    def unexpected_physics(*args, **kwargs):
        raise AssertionError("State access must not evaluate or advance physics")

    monkeypatch.setattr(mujoco, "mj_forward", unexpected_physics)
    monkeypatch.setattr(mujoco, "mj_step", unexpected_physics)
    current = state.snapshot()
    assert isinstance(current, JointState)
    assert current.qpos[0] == previous.qpos[0] + 0.2
    np.testing.assert_array_equal(state.get_frame_position("ee_site"), position)
    np.testing.assert_array_equal(current.bias_forces, previous.bias_forces)
    assert state.get_jacobian("ee_site").shape == (6, state.nv)
    assert state.get_mass_matrix().shape == (state.nv, state.nv)
    assert sim.data.time == 0
