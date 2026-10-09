import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import (
    RobotSpec,
    Simulator,
    create_environment,
)
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import Controller
from mujoco_lab.trajectory_generation import ControlTarget
from mujoco_lab.utils import Transform


def make_pair(right="forte", *, environment="empty", scene=None):
    return Simulator(
        create_environment(environment) if scene is None else scene,
        robots=[
            RobotSpec(
                "left",
                "forte",
                Transform(translation=[-0.8, 0, 0]),
                config=load_robot_config("forte"),
            ),
            RobotSpec(
                "right", right, Transform(translation=[0.8, 0, 0]), config=load_robot_config(right)
            ),
        ],
    )


@pytest.mark.parametrize("right,dimensions", [("forte", (18, 18, 16)), ("panda", (18, 18, 16))])
def test_composed_scene_has_distinct_bindings_and_simultaneous_homes(right, dimensions):
    sim = make_pair(right)
    model, data = sim.model, sim.data
    assert (model.nq, model.nv, model.nu) == dimensions
    left, other = sim.robots.values()
    assert left.model is other.model is model
    assert left.data is other.data is data
    for attribute in ["joint_ids", "qpos_indices", "dof_indices"]:
        assert not set(getattr(left.state, attribute)) & set(getattr(other.state, attribute))
    assert not set(left.actuator_ids) & set(other.actuator_ids)
    for robot in sim.robots.values():
        assert model.body(robot.state.root_body_id).name.startswith(robot.name + "/")
        key = model.key(robot.name + "/home")
        np.testing.assert_allclose(
            robot.state.snapshot().qpos, key.qpos[robot.state.qpos_indices], rtol=0, atol=1e-9
        )
        expected_ctrl = key.ctrl[robot.actuator_ids].copy()
        gripper = robot.gripper
        expected_ctrl[gripper.slot] = key.qpos[model.jnt_qposadr[gripper.joint_id]] * gripper.gear
        np.testing.assert_allclose(data.ctrl[robot.actuator_ids], expected_ctrl, rtol=0, atol=1e-9)
    if right == "panda":
        assert (other.state.nq, other.state.nv, other.num_actuators) == (9, 9, 8)
        finger1 = model.joint("right/panda_finger_joint1").id
        finger2 = model.joint("right/panda_finger_joint2").id
        assert set(model.eq_obj1id) | set(model.eq_obj2id) >= {finger1, finger2}
        assert finger2 not in model.actuator_trnid[:, 0]
    with pytest.raises(TypeError):
        sim.robots["extra"] = left


def test_canonical_mount_fallback_and_explicit_world_poses():
    sim = Simulator(
        create_environment("warehouse"),
        robots=[RobotSpec("arm", "forte", config=load_robot_config("forte"))],
    )
    robot = sim.robots["arm"]
    np.testing.assert_allclose(sim.data.body(robot.state.root_body_id).xpos, [0, 0, 0.858])
    pose = Transform.from_pose_mmdeg([100, 200, 300, 20, -15, 35])
    for environment in ["empty", "warehouse"]:
        explicit = Simulator(
            create_environment(environment),
            robots=[RobotSpec("arm", "forte", pose, config=load_robot_config("forte"))],
        )
        base = explicit.data.body(explicit.robots["arm"].state.root_body_id)
        np.testing.assert_allclose(
            base.xpos,
            pose.as_translation() + pose.as_rotation().as_matrix() @ [0, 0, 0.058],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            base.xmat.reshape(3, 3), pose.as_rotation().as_matrix(), atol=1e-14
        )


def test_robot_state_snapshots_are_fresh_and_owned():
    sim = make_pair()
    left, right = sim.robots.values()
    access = left.state
    assert access.model is sim.model and access.data is sim.data
    old = access.snapshot()
    old_qpos = old.qpos.copy()
    assert not np.shares_memory(old.qpos, sim.data.qpos)
    assert not np.shares_memory(old.qvel, sim.data.qvel)
    assert not np.shares_memory(old.bias_forces, sim.data.qfrc_bias)
    bias = access.snapshot().bias_forces
    np.testing.assert_array_equal(bias, sim.data.qfrc_bias[access.dof_indices])
    bias[:] = 99
    np.testing.assert_array_equal(old.bias_forces, access.snapshot().bias_forces)
    old.qpos[:] = 99
    np.testing.assert_array_equal(left.state.snapshot().qpos, old_qpos)
    saved = right.state.snapshot()
    sim.step()
    np.testing.assert_allclose(
        saved.qpos, sim.model.key("right/home").qpos[right.state.qpos_indices], rtol=0, atol=1e-9
    )
    assert left.state.snapshot().time == sim.data.time
    assert left.state is access
    sim.reset()
    assert left.state is access
    assert access.snapshot().time == 0
    np.testing.assert_array_equal(access.snapshot().qpos, old_qpos)


class ConstantController(Controller):
    def __init__(self, values, output_kind="torque"):
        self.values = values
        self.output_kind = output_kind

    def compute(self, state, target):
        return self.values


def test_scoped_input_clipping_and_one_shared_step(monkeypatch):
    sim = make_pair("panda")
    left, right = sim.robots.values()
    sentinel = sim.data.ctrl[right.actuator_ids].copy()
    sentinel[0] += 0.01
    sim.data.ctrl[right.actuator_ids] = sentinel
    left.change_controller(ConstantController(np.full(7, 1000.0)))
    left.control()
    np.testing.assert_array_equal(sim.data.ctrl[right.actuator_ids], sentinel)
    right_times = []
    compute = right.controller.compute

    def record_right_control(state, target):
        right_times.append(state.time)
        return compute(state, target)

    monkeypatch.setattr(right.controller, "compute", record_right_control)
    previous = right.joint_state
    sim.run_steps(3)
    assert right.joint_state is not previous
    for robot in sim.robots.values():
        current = robot.state.snapshot()
        assert robot.joint_state.time == current.time == sim.data.time
        np.testing.assert_array_equal(robot.joint_state.qpos, current.qpos)
        np.testing.assert_array_equal(robot.joint_state.qvel, current.qvel)
        np.testing.assert_array_equal(robot.joint_state.bias_forces, current.bias_forces)
    np.testing.assert_allclose(right_times, np.arange(3) * sim.dt)
    arm_ids = np.asarray(left.actuator_ids)[left.control_actuator_slots]
    np.testing.assert_array_equal(sim.data.ctrl[arm_ids], sim.model.actuator_ctrlrange[arm_ids, 1])
    assert sim.data.time == pytest.approx(3 * sim.model.opt.timestep)
    sim.step()
    assert sim.data.time == pytest.approx(4 * sim.dt)
    np.testing.assert_allclose(right_times, np.arange(4) * sim.dt)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.zeros(8), "joint commands"),
        (np.ones(1), "joint commands"),
        (np.full(7, np.nan), "finite actuator inputs"),
    ],
)
def test_later_invalid_command_does_not_advance_physics(bad, message):
    sim = make_pair()
    left, right = sim.robots.values()
    left.change_controller(ConstantController(np.ones(7)))
    right.change_controller(ConstantController(bad))
    before = sim.data.ctrl.copy()
    expected_left = before[left.actuator_ids].copy()
    arm_ids = [left.actuator_ids[slot] for slot in left.control_actuator_slots]
    expected_left[left.control_actuator_slots] = 1 / sim.model.actuator_gear[arm_ids, 0]
    with pytest.raises(ValueError, match=message):
        sim.step()
    np.testing.assert_array_equal(sim.data.ctrl[left.actuator_ids], expected_left)
    np.testing.assert_array_equal(sim.data.ctrl[right.actuator_ids], before[right.actuator_ids])
    assert sim.data.time == 0
    assert sim._state.get_state() == "idle"
    assert not sim._stop_requested

    clean = make_pair()
    clean_left = clean.robots["left"]
    clean_left.change_controller(ConstantController(np.ones(7)))
    clean_expected = clean.data.ctrl[clean_left.actuator_ids].copy()
    clean_arm_ids = [clean_left.actuator_ids[slot] for slot in clean_left.control_actuator_slots]
    clean_expected[clean_left.control_actuator_slots] = (
        1 / clean.model.actuator_gear[clean_arm_ids, 0]
    )
    clean.step()
    np.testing.assert_array_equal(clean.data.ctrl[clean_left.actuator_ids], clean_expected)


def test_controller_gains_and_targets_are_not_shared_and_reset_keeps_configuration():
    sim = make_pair()
    initial_qpos = sim.data.qpos.copy()
    initial_ctrl = sim.data.ctrl.copy()
    left, right = sim.robots.values()
    left.change_controller(create_test_controller(left, controller="pd"))
    active_controller = left.controller
    active_target = left.target
    active_mapping = (
        left.control_joint_names,
        left.control_qpos_indices,
        left.control_joint_slots,
    )
    replacement = create_test_controller(left, controller="pd")
    assert replacement._owner is left.state
    assert left.controller is active_controller and left.target is active_target
    assert active_mapping == (
        left.control_joint_names,
        left.control_qpos_indices,
        left.control_joint_slots,
    )
    np.testing.assert_array_equal(sim.data.ctrl, initial_ctrl)
    another = create_test_controller(right, controller="pd")
    old_gain = another.kp.copy()
    left.controller.kp[0] += 5
    left.target = ControlTarget(left.target.position + 0.02)
    np.testing.assert_array_equal(another.kp, old_gain)
    right.change_controller(create_test_controller(right, controller="computed_torque"))
    right.controller.kp = 850
    sim.run_steps(50)
    assert sim.data.time == pytest.approx(50 * sim.dt)
    np.testing.assert_array_equal(right.target.position, right.reference_generator.state().position)
    gains = left.controller.kp.copy()
    sim.reset()
    assert sim.data.time == 0 and right.controller.get_tracking_error() == 0
    np.testing.assert_array_equal(left.controller.kp, gains)
    np.testing.assert_array_equal(left.target.position, left.joint_state.qpos[:7])
    assert right.controller.kp == 850
    np.testing.assert_array_equal(sim.data.qpos, initial_qpos)
    np.testing.assert_array_equal(sim.data.ctrl, initial_ctrl)


def test_dynamics_are_robot_scoped_and_buffers_do_not_alias_other_robots():
    sim = make_pair()
    left, right = sim.robots.values()
    jac = left.state.get_jacobian("grasp")
    saved_jac = jac.copy()
    mass = left.state.get_mass_matrix()
    saved_mass = mass.copy()
    right.state.get_jacobian("grasp")
    right.state.get_mass_matrix()
    np.testing.assert_array_equal(jac, saved_jac)
    np.testing.assert_array_equal(mass, saved_mass)
    whole = np.zeros((sim.model.nv, sim.model.nv))
    mujoco.mj_fullM(sim.model, sim.data, whole)
    for robot in sim.robots.values():
        expected = whole[np.ix_(robot.state.dof_indices, robot.state.dof_indices)]
        np.testing.assert_array_equal(robot.state.get_mass_matrix(), expected)
        assert robot.state.get_jacobian("grasp").shape == (6, robot.state.nv)
    np.testing.assert_array_equal(whole[np.ix_(left.state.dof_indices, right.state.dof_indices)], 0)
    with pytest.raises(KeyError, match="right/grasp"):
        left.state.get_frame_position("right/grasp")


def test_environment_prefix_and_free_joint_do_not_corrupt_ownership():
    def environment(name):
        spec = create_environment(name)
        body = spec.worldbody.add_body(name="left/helper", pos=[0, 2, 2])
        body.add_freejoint(name="environment_free")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0])
        body.add_site(name="left/helper_site")
        return spec

    sim = make_pair("panda", scene=environment("empty"))
    left = sim.robots["left"]
    assert left.state.joint_ids[0] == 1
    assert left.state.qpos_indices[0] == 7
    assert left.state.dof_indices[0] == 6
    assert left.actuator_ids[0] == 0
    assert sim.model.body("left/helper").id not in sim.model.jnt_bodyid[left.state.joint_ids]
    with pytest.raises(KeyError, match="helper_site"):
        left.state.get_frame_position("helper_site")
    left.change_controller(create_test_controller(left, controller="computed_torque"))
    sim.step()
    assert left.state.get_jacobian("grasp").shape == (6, 9)
