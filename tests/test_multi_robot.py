import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import pytest

from mujoco_lab import (
    RobotSpec,
    Simulator,
    SimulatorManager,
    create_environment,
)
from mujoco_lab.control import Controller, ControlTarget, create_controller
from mujoco_lab.utils import Transform


def make_pair(right="forte", *, environment="empty", scene=None):
    return Simulator(
        create_environment(environment) if scene is None else scene,
        robots=[
            RobotSpec("left", "forte", Transform(translation=[-0.8, 0, 0])),
            RobotSpec("right", right, Transform(translation=[0.8, 0, 0])),
        ],
    )


def manager_with(simulator, name="simulator"):
    manager = SimulatorManager()
    manager.add_simulator(name, simulator)
    return manager


def passive_viewer(*, on_lock=None, on_sync=None):
    viewer = Mock()
    viewer.is_running.side_effect = [True, False]

    def lock():
        if on_lock is not None:
            on_lock()
        return nullcontext()

    def sync(*, state_only=False):
        assert state_only
        if on_sync is not None:
            on_sync()

    viewer.lock.side_effect = lock
    viewer.sync.side_effect = sync
    viewer._sim = lambda: None
    return viewer


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
        np.testing.assert_array_equal(
            robot.state.snapshot().qpos, key.qpos[robot.state.qpos_indices]
        )
        np.testing.assert_array_equal(data.ctrl[robot.actuator_ids], key.ctrl[robot.actuator_ids])
    if right == "panda":
        assert (other.state.nq, other.state.nv, other.nu) == (9, 9, 8)
        finger1 = model.joint("right/panda_finger_joint1").id
        finger2 = model.joint("right/panda_finger_joint2").id
        assert set(model.eq_obj1id) | set(model.eq_obj2id) >= {finger1, finger2}
        assert finger2 not in model.actuator_trnid[:, 0]
    with pytest.raises(TypeError):
        sim.robots["extra"] = left


def test_canonical_mount_fallback_and_explicit_world_poses():
    sim = Simulator(create_environment("warehouse"), robots=[RobotSpec("arm", "forte")])
    robot = sim.robots["arm"]
    np.testing.assert_allclose(sim.data.body(robot.state.root_body_id).xpos, [0, 0, 0.858])
    pose = Transform.from_pose_mmdeg([100, 200, 300, 20, -15, 35])
    for environment in ["empty", "warehouse"]:
        explicit = Simulator(
            create_environment(environment), robots=[RobotSpec("arm", "forte", pose)]
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


def test_missing_canonical_mount_is_rejected_by_native_attachment():
    scene = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")

    with pytest.raises(ValueError, match="One of frame or site must be specified"):
        Simulator(scene, robots=[RobotSpec("arm", "forte")])

    assert scene.body("arm/base_link") is None


def test_instance_names_and_missing_multi_robot_poses_are_rejected():
    pose = Transform.identity()
    for robots in [
        [RobotSpec("", "forte")],
        [RobotSpec("a/b", "forte")],
        [RobotSpec("same", "forte", pose), RobotSpec("same", "panda", pose)],
        [RobotSpec("a", "forte"), RobotSpec("b", "forte", pose)],
    ]:
        with pytest.raises(ValueError):
            Simulator(create_environment("empty"), robots=robots)


def test_zero_steps_and_empty_environment_do_not_evaluate_control():
    sim = make_pair()
    before = sim.data.qpos.copy()
    stats = sim.run_steps(0)
    assert all(value.steps == 0 for value in stats.values())
    assert sim.data.time == 0
    np.testing.assert_array_equal(sim.data.qpos, before)
    assert Simulator(create_environment("empty")).run_steps(3) == {}


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
    np.testing.assert_array_equal(
        saved.qpos, sim.model.key("right/home").qpos[right.state.qpos_indices]
    )
    assert left.state.snapshot().time == sim.data.time
    assert left.state is access
    sim.reset()
    assert left.state is access
    assert access.snapshot().time == 0
    np.testing.assert_array_equal(access.snapshot().qpos, old_qpos)


class ConstantController(Controller):
    def __init__(self, values):
        self.values = values

    def compute(self, state, target):
        return self.values


def test_robot_update_state_refreshes_owned_snapshot_without_forward_or_step(monkeypatch):
    sim = make_pair("panda")
    left, right = sim.robots.values()
    previous, other = left.joint_state, right.joint_state
    initial = previous.qpos.copy()
    sim.data.qpos[left.state.qpos_indices] += 0.1
    sim.data.qvel[left.state.dof_indices] = 0.2
    sim.data.qfrc_bias[left.state.dof_indices] = 0.3
    sim.data.time = 0.5

    def unexpected_physics(*args):
        raise AssertionError("Updating a Robot snapshot must not evaluate physics")

    with monkeypatch.context() as patch:
        patch.setattr(mujoco, "mj_forward", unexpected_physics)
        patch.setattr(mujoco, "mj_step", unexpected_physics)
        assert left.update_state() is None
    current = left.joint_state
    assert current is not previous
    assert right.joint_state is other
    assert current.time == sim.data.time == 0.5
    np.testing.assert_array_equal(current.qpos, initial + 0.1)
    np.testing.assert_array_equal(current.qvel, 0.2)
    np.testing.assert_array_equal(current.bias_forces, 0.3)
    np.testing.assert_array_equal(previous.qpos, initial)
    sim.reset()
    assert left.joint_state.time == right.joint_state.time == 0
    np.testing.assert_array_equal(left.joint_state.qpos, initial)
    np.testing.assert_array_equal(current.qpos, initial + 0.1)


def test_robot_control_uses_cached_state_and_applies_only_its_inputs():
    sim = make_pair("panda")
    left, right = sim.robots.values()
    states = []

    class RecordingController(ConstantController):
        def compute(self, state, target):
            states.append(state)
            return super().compute(state, target)

    left.change_controller(RecordingController(np.full(left.nu, 1000.0)))
    sim.data.qpos[left.state.qpos_indices] += 0.1
    sim.data.time = 0.5
    positions = sim.data.qpos.copy()
    other_inputs = sim.data.ctrl[right.actuator_ids].copy()
    cached = left.joint_state
    assert left.control()
    assert len(states) == 1 and states[0] is cached
    assert left.joint_state is cached
    left.update_state()
    assert left.control()
    assert states[-1] is left.joint_state and left.joint_state is not cached
    assert left.joint_state.time == sim.data.time == 0.5
    np.testing.assert_array_equal(left.joint_state.qpos, positions[left.state.qpos_indices])
    np.testing.assert_array_equal(sim.data.qpos, positions)
    np.testing.assert_array_equal(
        sim.data.ctrl[left.actuator_ids], sim.model.actuator_ctrlrange[left.actuator_ids, 1]
    )
    np.testing.assert_array_equal(sim.data.ctrl[right.actuator_ids], other_inputs)
    inputs = sim.data.ctrl.copy()
    left.change_controller(None)
    np.testing.assert_array_equal(sim.data.ctrl, inputs)
    assert len(states) == 2


@pytest.mark.parametrize("asset", ["panda"])
def test_robot_accepts_custom_controllers_without_forte_specific_checks(asset):
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("arm", asset)])
    robot = sim.robots["arm"]
    command = sim.data.ctrl[robot.actuator_ids].copy()
    command[0] += 0.01
    robot.change_controller(ConstantController(command))
    assert robot.controller.tracking_error is None
    stats = sim.step()[robot.name]
    np.testing.assert_array_equal(sim.data.ctrl[robot.actuator_ids], command)
    assert stats.steps == 1
    assert stats.errors == []
    assert not sim.data.warning.number.any()


def test_scoped_input_clipping_stats_and_one_shared_step():
    sim = make_pair("panda")
    left, right = sim.robots.values()
    sentinel = sim.data.ctrl[right.actuator_ids].copy()
    sentinel[0] += 0.01
    sim.data.ctrl[right.actuator_ids] = sentinel
    left.change_controller(ConstantController(np.full(left.nu, 1000.0)))
    previous = right.joint_state
    stats = sim.run_steps(3)
    assert right.joint_state is not previous
    assert right.joint_state.time == pytest.approx(2 * sim.model.opt.timestep)
    np.testing.assert_array_equal(sim.data.ctrl[right.actuator_ids], sentinel)
    np.testing.assert_array_equal(
        sim.data.ctrl[left.actuator_ids], sim.model.actuator_ctrlrange[left.actuator_ids, 1]
    )
    assert sim.data.time == pytest.approx(3 * sim.model.opt.timestep)
    assert stats["left"].saturated_steps == 3
    assert stats["right"].saturated_steps == 0
    assert stats["left"].steps == stats["right"].steps == 3
    assert stats["right"].errors == []
    following = sim.step()
    assert following["left"].steps == 1 and stats["left"].steps == 3


@pytest.mark.parametrize("bad", [np.zeros(7), np.full(8, np.nan)])
def test_later_invalid_command_does_not_advance_physics(bad):
    sim = make_pair()
    left, right = sim.robots.values()
    left.change_controller(ConstantController(np.ones(left.nu)))
    right.change_controller(ConstantController(bad))
    before = sim.data.ctrl.copy()
    with pytest.raises(ValueError, match="finite actuator inputs"):
        sim.step()
    np.testing.assert_array_equal(sim.data.ctrl[left.actuator_ids], [*np.ones(7), 0])
    np.testing.assert_array_equal(sim.data.ctrl[right.actuator_ids], before[right.actuator_ids])
    assert sim.data.time == 0
    assert sim._state.state == "running"

    clean = make_pair()
    clean_left = clean.robots["left"]
    clean_left.change_controller(ConstantController(np.ones(left.nu)))
    clean.step()
    np.testing.assert_array_equal(clean.data.ctrl[clean_left.actuator_ids], [*np.ones(7), 0])


def test_controller_gains_and_targets_are_not_shared_and_reset_keeps_configuration():
    sim = make_pair()
    initial_qpos = sim.data.qpos.copy()
    initial_ctrl = sim.data.ctrl.copy()
    left, right = sim.robots.values()
    left.change_controller(create_controller("pd", left))
    another = create_controller("pd", right)
    old_gain = another.kp.copy()
    left.controller.kp[0] += 5
    left.target = ControlTarget(left.target.position + 0.02)
    np.testing.assert_array_equal(another.kp, old_gain)
    right.change_controller(create_controller("osc", right))
    right.controller.task_kp = 850
    stats = sim.run_steps(50)
    assert right.controller._target is not None
    gains = left.controller.kp.copy()
    sim.reset()
    assert sim.data.time == 0 and right.controller._target is None
    np.testing.assert_array_equal(left.controller.kp, gains)
    np.testing.assert_array_equal(left.target.position, left.joint_state.qpos[:7])
    assert right.controller.task_kp == 850
    assert all(value.steps == 50 for value in stats.values())
    np.testing.assert_array_equal(sim.data.qpos, initial_qpos)
    np.testing.assert_array_equal(sim.data.ctrl, initial_ctrl)


def test_dynamics_are_robot_scoped_and_buffers_do_not_alias_other_robots():
    sim = make_pair()
    left, right = sim.robots.values()
    jac = left.state.get_jacobian("ee_site")
    saved_jac = jac.copy()
    mass = left.state.get_mass_matrix()
    saved_mass = mass.copy()
    right.state.get_jacobian("ee_site")
    right.state.get_mass_matrix()
    np.testing.assert_array_equal(jac, saved_jac)
    np.testing.assert_array_equal(mass, saved_mass)
    whole = np.zeros((sim.model.nv, sim.model.nv))
    mujoco.mj_fullM(sim.model, sim.data, whole)
    for robot in sim.robots.values():
        expected = whole[np.ix_(robot.state.dof_indices, robot.state.dof_indices)]
        np.testing.assert_array_equal(robot.state.get_mass_matrix(), expected)
        assert robot.state.get_jacobian("ee_site").shape == (6, robot.state.nv)
    np.testing.assert_array_equal(whole[np.ix_(left.state.dof_indices, right.state.dof_indices)], 0)
    with pytest.raises(ValueError, match="no site"):
        left.state.get_frame_position("right/ee_site")


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
    with pytest.raises(ValueError, match="no site"):
        left.state.get_frame_position("helper_site")
    left.change_controller(create_controller("osc", left))
    sim.step()
    assert left.state.get_jacobian("ee_site").shape == (6, 9)


@pytest.mark.parametrize(
    "kind,entity_name",
    [
        ("body", "left/base_link"),
        ("geom", "left/forearm_collision"),
        ("site", "left/ee_site"),
    ],
)
def test_mujoco_rejects_environment_name_collisions(kind, entity_name):
    def environment(name):
        spec = create_environment(name)
        if kind == "body":
            spec.worldbody.add_body(name=entity_name)
        elif kind == "geom":
            spec.worldbody.add_geom(
                name=entity_name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0]
            )
        else:
            spec.worldbody.add_site(name=entity_name)
        return spec

    with pytest.raises(ValueError, match=f"repeated name '{entity_name}' in {kind}"):
        make_pair(scene=environment("empty"))


def test_environment_home_activation_and_mocap_survive_robot_initialization_and_reset():
    def environment(name):
        spec = create_environment(name)
        body = spec.worldbody.add_body(name="fixture", pos=[0, 2, 1])
        body.add_joint(name="fixture_joint", type=mujoco.mjtJoint.mjJNT_HINGE)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0])
        spec.add_actuator(
            name="fixture_motor",
            target="fixture_joint",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_FILTER,
            dynprm=[0.1] + [0.0] * 9,
        )
        marker = spec.worldbody.add_body(name="marker", mocap=True)
        marker.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0])
        spec.add_key(
            name="home",
            time=2,
            qpos=[0.7],
            qvel=[0.3],
            act=[0.4],
            ctrl=[0.6],
            mpos=[1, 2, 3],
            mquat=[1, 0, 0, 0],
        )
        return spec

    sim = make_pair("panda", scene=environment("empty"))
    expected = sim.data.qpos.copy()
    assert sim.data.time == 2
    assert sim.data.qpos[0] == 0.7 and sim.data.qvel[0] == 0.3
    np.testing.assert_allclose(sim.data.act, [0.4])
    np.testing.assert_allclose(sim.data.ctrl[0], 0.6)
    np.testing.assert_array_equal(sim.data.mocap_pos, [[1, 2, 3]])
    sim.data.qpos[:] = 0
    sim.data.ctrl[:] = 0
    sim.data.act[:] = 0
    sim.data.mocap_pos[:] = 0
    sim.reset()
    np.testing.assert_array_equal(sim.data.qpos, expected)
    np.testing.assert_allclose(sim.data.act, [0.4])
    np.testing.assert_allclose(sim.data.ctrl[0], 0.6)
    np.testing.assert_array_equal(sim.data.mocap_pos, [[1, 2, 3]])


def test_pd_annotations_share_scratch_and_preserve_scene_state():
    from mujoco_lab.rendering import annotations

    sim = make_pair()
    left, right = sim.robots.values()
    for robot, offset in [(left, 0.15), (right, -0.25)]:
        controller = create_controller("pd", robot)
        robot.change_controller(controller)
        target = robot.target.position.copy()
        target[0] += offset
        robot.target = ControlTarget(target)
    sim.run_steps(10)
    before = {
        field: getattr(sim.data, field).copy()
        for field in ["qpos", "qvel", "ctrl", "site_xpos", "xpos", "qfrc_bias"]
    }
    mass_before = np.empty((sim.model.nv, sim.model.nv))
    mujoco.mj_fullM(sim.model, sim.data, mass_before)
    targets, expected_targets, expected_actual = [], [], []
    for robot in sim.robots.values():
        reference = mujoco.MjData(sim.model)
        mujoco.mj_copyData(reference, sim.model, sim.data)
        reference.qpos[robot.state.qpos_indices[:7]] = robot.target.position
        mujoco.mj_kinematics(sim.model, reference)
        site = robot.state.site_id("ee_site")
        expected_targets.append(reference.site_xpos[site].copy())
        expected_actual.append(sim.data.site_xpos[site].copy())
        targets.append(robot.target.position.copy())

    scene = mujoco.MjvScene(sim.model, maxgeom=100)
    scene.ngeom = 0
    with (
        patch.object(mujoco, "MjData", wraps=mujoco.MjData) as allocate,
        patch.object(mujoco, "mj_copyData", wraps=mujoco.mj_copyData) as copy_data,
    ):
        annotations.annotate(scene, sim, sim.data)
    assert allocate.call_count == copy_data.call_count == 1
    scratch = copy_data.call_args.args[0]
    assert scratch is not sim.data
    np.testing.assert_array_equal(
        scratch.qpos[left.state.qpos_indices], sim.data.qpos[left.state.qpos_indices]
    )
    markers = list(scene.geoms[: scene.ngeom])
    for rgba, expected in [
        (annotations.TARGET_RGBA, expected_targets),
        (annotations.ACTUAL_RGBA, expected_actual),
    ]:
        actual = [geom.pos for geom in markers if np.allclose(geom.rgba, rgba)]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    for field, values in before.items():
        np.testing.assert_array_equal(getattr(sim.data, field), values)
    mass_after = np.empty_like(mass_before)
    mujoco.mj_fullM(sim.model, sim.data, mass_after)
    np.testing.assert_array_equal(mass_after, mass_before)
    for robot, target in zip(sim.robots.values(), targets):
        np.testing.assert_array_equal(robot.target.position, target)


def test_render_is_observational_and_handles_two_controllers(tmp_path):
    code = r"""
import sys
import mujoco
import numpy as np
from mujoco_lab import Simulator, RobotSpec, SimulatorManager, create_environment
from mujoco_lab.control import create_controller
from mujoco_lab.utils import Transform

def make():
    specs=[RobotSpec(n,"forte",Transform(translation=[x,0,0]))
           for n,x in [("left",-.8),("right",.8)]]
    sim=Simulator(create_environment("empty"), robots=specs)
    for name,mode in [("left","pd"),("right","osc")]:
        sim.robots[name].change_controller(create_controller(mode,sim.robots[name]))
    return sim

sim, reference = make(), make()
manager=SimulatorManager()
manager.add_simulator("controlled",sim)
osc=sim.robots["right"].controller
assert osc._target is None
initial_targets={name:robot.target.position.copy() for name,robot in sim.robots.items()}
updates=[]
sim.target_updater=lambda current: updates.append(current.data.time)
manager.save_frame("controlled",sys.argv[1]+"/zero.png")
assert osc._target is None
assert updates == []
for name,robot in sim.robots.items():
    np.testing.assert_array_equal(robot.target.position,initial_targets[name])
for _ in range(10):
    sim.step(); reference.step()
    qpos=sim.data.qpos.copy(); ctrl=sim.data.ctrl.copy()
    caches=(osc._force.copy(),osc._target.copy(),osc.tracking_error,
            sim.robots["right"].target.position.copy())
    manager.save_frame("controlled",sys.argv[1]+"/controlled.png")
    np.testing.assert_array_equal(sim.data.qpos,qpos)
    np.testing.assert_array_equal(sim.data.ctrl,ctrl)
    for actual,expected in zip((osc._force,osc._target,osc.tracking_error,
                                sim.robots["right"].target.position),caches):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(sim.data.qpos,reference.data.qpos)
    np.testing.assert_array_equal(sim.data.ctrl,reference.data.ctrl)
sim.step(); reference.step()
np.testing.assert_array_equal(sim.data.qpos,reference.data.qpos)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={**os.environ, "MUJOCO_GL": "egl"},
        cwd=tmp_path,
        check=True,
    )
    assert (tmp_path / "controlled.png").stat().st_size > 1000


def test_save_frame_failure_retains_rendering_state(tmp_path):
    sim = make_pair()
    manager = manager_with(sim)
    before = sim.data.qpos.copy()
    with (
        patch.object(mujoco, "Renderer", side_effect=RuntimeError("render failed")),
        pytest.raises(RuntimeError, match="render failed"),
    ):
        manager.save_frame("simulator", tmp_path / "failed.png")
    np.testing.assert_array_equal(sim.data.qpos, before)
    assert sim._state.state == "rendering"


@pytest.mark.parametrize("layout", ["dual_forte", "forte_panda"])
def test_multi_robot_example(layout, tmp_path):
    example = Path(__file__).parents[1] / "examples/multi_robot.py"
    result = subprocess.run(
        [sys.executable, str(example), "--layout", layout, "--headless", "--steps", "100"],
        cwd=tmp_path,
        env={**os.environ, "MUJOCO_GL": "disable"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Shared simulated seconds: 0.200" in result.stdout
    assert "left (forte)" in result.stdout and "right (" in result.stdout


def test_change_controller_preserves_binding_and_resets_replacement_history():
    sim = make_pair()
    manager = manager_with(sim)
    left, right = sim.robots.values()
    previous = create_controller("pd", right)
    right.change_controller(previous)
    osc = create_controller("osc", left)
    assert osc.robot_state is left.state
    with pytest.raises(ValueError, match="only one Robot"):
        right.change_controller(osc)
    assert right.controller is previous
    pd = create_controller("pd", left)
    with pytest.raises(ValueError, match="only one Robot"):
        right.change_controller(pd)
    assert right.controller is previous
    left.change_controller(pd)
    pd.tracking_error = 4
    left.change_controller(pd)
    assert pd.tracking_error == 4
    left.change_controller(None)
    left.change_controller(pd)
    assert pd.tracking_error == 0

    def inspect_open_scope():
        with pytest.raises(RuntimeError, match="busy with viewing"):
            left.change_controller(None)

    with patch(
        "mujoco.viewer.launch_passive", return_value=passive_viewer(on_lock=inspect_open_scope)
    ):
        manager.show("simulator")
    assert left.controller is pd


def test_headless_control_owns_inputs_and_retains_failure_state(monkeypatch):
    sim = make_pair("panda")
    left = sim.robots["left"]
    left.change_controller(ConstantController(np.ones(left.nu)))
    panda_input = sim.data.ctrl[sim.robots["right"].actuator_ids].copy()
    sim.step()
    np.testing.assert_array_equal(sim.data.ctrl[left.actuator_ids], [*np.ones(7), 0])
    np.testing.assert_array_equal(sim.data.ctrl[sim.robots["right"].actuator_ids], panda_input)

    def fail_step(model, data):
        assert sim._state.state == "running"
        raise RuntimeError("step failed")

    with monkeypatch.context() as patch:
        patch.setattr(mujoco, "mj_step", fail_step)
        with pytest.raises(RuntimeError, match="step failed"):
            sim.step()
    assert sim._state.state == "running"


def test_native_reset_keeps_controller_history_until_programmatic_reset():
    sim = make_pair("panda")
    initial_qpos = sim.data.qpos.copy()
    manager = manager_with(sim)
    robot = sim.robots["left"]
    robot.change_controller(create_controller("osc", robot))
    sim.step()
    target = robot.target.position.copy()
    cached_target = robot.controller._target.copy()

    def native_reset():
        mujoco.mj_resetData(sim.model, sim.data)
        np.testing.assert_array_equal(sim.data.qpos, sim.model.qpos0)
        np.testing.assert_array_equal(sim.data.ctrl, 0)
        np.testing.assert_array_equal(robot.target.position, target)
        np.testing.assert_array_equal(robot.controller._target, cached_target)

    with patch("mujoco.viewer.launch_passive", return_value=passive_viewer(on_sync=native_reset)):
        manager.show("simulator")
    assert not np.array_equal(sim.data.qpos, initial_qpos)
    sim.reset()
    assert robot.controller._target is None
    np.testing.assert_array_equal(sim.data.qpos, initial_qpos)


def test_contact_solver_still_sees_both_robots():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec(name, "forte", Transform.identity()) for name in ["left", "right"]],
    )
    left, right = sim.robots.values()
    contacts = [
        {
            sim.model.geom(c.geom1).name.partition("/")[0],
            sim.model.geom(c.geom2).name.partition("/")[0],
        }
        for c in sim.data.contact
    ]
    assert {left.name, right.name} in contacts
