"""OGBench-layout cubes and a real Panda grasp, lift, and released stack."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.control import create_controller
from mujoco_lab.rendering.annotations import draw_joint_torques
from mujoco_lab.tasks import CubeStackExpert, CubeStackMotionGenerator, CubeStackTask

STARTS = {
    2: ((0.425, -0.2, 0.82), (0.425, 0.2, 0.82)),
    3: ((0.35, -0.1, 0.82), (0.5, -0.2, 0.82), (0.5, 0, 0.82)),
    4: ((0.5, -0.05, 0.82), (0.5, -0.2, 0.82), (0.35, -0.2, 0.82), (0.35, -0.05, 0.82)),
}
GOALS = {
    2: ((0.425, 0, 0.82), (0.425, 0, 0.86)),
    3: ((0.425, 0.2, 0.82), (0.425, 0.2, 0.86), (0.425, 0.2, 0.90)),
    4: ((0.425, 0.2, 0.82), (0.425, 0.2, 0.86), (0.425, 0.2, 0.90), (0.425, 0.2, 0.94)),
}
COLORS = ((0.96, 0.26, 0.33), (0.35, 0.55, 0.91), (1.0, 0.69, 0.21), (0.06, 0.74, 0.21))


def make_panda_expert(task):
    robot = task.simulator.robots["panda"]
    robot.change_controller(
        create_controller("position", robot, gravity_compensation=True, frame="grasp")
    )
    return CubeStackExpert(task, CubeStackMotionGenerator(task))


def place_cubes_at_goals(simulator, task):
    for index, goal in enumerate(task.goals):
        joint = simulator.model.joint(f"cube{index}/object_joint_0")
        simulator.data.qpos[int(joint.qposadr[0]) : int(joint.qposadr[0]) + 3] = goal
    mujoco.mj_forward(simulator.model, simulator.data)


def test_task_succeeds_from_physical_stack_without_executor():
    simulator = Simulator(create_cube_stack(2))
    updates = []

    def updater(sim):
        updates.append(sim.data.time)

    simulator.target_updater = updater
    task = CubeStackTask(simulator, 2)
    assert simulator.target_updater is updater
    assert not simulator.robots
    place_cubes_at_goals(simulator, task)
    assert task.status().ogbench_success
    assert not task.status().released_stable_stack

    for _ in range(400):
        task.status()
        simulator.run_steps(1)
    assert updates
    assert task.status().released_stable_stack
    assert task.status().support_contacts

    joint = simulator.model.joint("cube1/object_joint_0")
    simulator.data.qpos[int(joint.qposadr[0])] += 0.03
    mujoco.mj_forward(simulator.model, simulator.data)
    assert task.status().ogbench_success
    assert not task.status().released_stable_stack
    simulator.data.qpos[int(joint.qposadr[0])] -= 0.03
    mujoco.mj_forward(simulator.model, simulator.data)
    simulator.data.qvel[int(joint.dofadr[0])] = 0.1
    assert not task.status().released_stable_stack
    simulator.data.qvel[int(joint.dofadr[0])] = 0
    simulator.data.qpos[int(joint.qposadr[0]) + 2] += 0.1
    mujoco.mj_forward(simulator.model, simulator.data)
    assert not task.status().support_contacts
    assert not task.status().released_stable_stack
    task.reset()
    assert simulator.target_updater is updater
    assert not task.status().ogbench_success


def test_task_rejects_robot_contact_even_after_stable_hold(tmp_path, monkeypatch):
    path = tmp_path / "blocker.xml"
    path.write_text("""
        <mujoco><worldbody><body name="base">
          <geom name="touch" type="sphere" pos=".425 0 2" size=".02"/>
        </body></worldbody></mujoco>
    """)
    monkeypatch.setitem(ROBOT_ASSETS, "blocker", RobotAsset(path))
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("blocker", "blocker")])
    task = CubeStackTask(simulator, 2)
    place_cubes_at_goals(simulator, task)
    for _ in range(400):
        task.status()
        simulator.run_steps(1)
    assert task.status().released_stable_stack

    simulator.model.geom("blocker/touch").pos[:] = [0.425, 0, 0.02]
    mujoco.mj_forward(simulator.model, simulator.data)
    assert any(
        "blocker/touch"
        in (simulator.model.geom(contact.geom1).name, simulator.model.geom(contact.geom2).name)
        for contact in simulator.data.contact
    )
    assert task.status().ogbench_success
    assert not task.status().released_stable_stack


@pytest.mark.parametrize("cubes", [2, 3, 4])
@pytest.mark.parametrize("furniture", ["table_shelf", "warehouse"])
def test_xml_cube_environments_compile_with_task_5_layout(cubes, furniture):
    simulator = Simulator(create_cube_stack(cubes, environment=furniture))
    model, data = simulator.model, simulator.data
    np.testing.assert_allclose(data.site("robot_mount").xpos, [0, 0, 0.8])
    assert (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "small_shelf/box") >= 0) == (
        furniture == "warehouse"
    )
    for index in range(cubes):
        prefix = f"cube{index}/"
        body = model.body(prefix + "object_0")
        target = model.body(prefix + "object_target_0")
        joint = model.joint(prefix + "object_joint_0")
        cube_geom = model.geom(prefix + "object_0")
        target_geom = model.geom(prefix + "target_object_0")
        assert joint.type == mujoco.mjtJoint.mjJNT_FREE
        assert target.mocapid >= 0
        np.testing.assert_allclose(data.body(body.id).xpos, STARTS[cubes][index])
        np.testing.assert_allclose(data.body(target.id).xpos, GOALS[cubes][index])
        np.testing.assert_allclose(cube_geom.size, [0.02, 0.02, 0.02])
        np.testing.assert_allclose(cube_geom.rgba, [*COLORS[index], 1])
        np.testing.assert_allclose(target_geom.rgba, [*COLORS[index], 0.2])
        assert target_geom.contype == target_geom.conaffinity == 0


@pytest.mark.parametrize("cubes", [2, 3, 4])
def test_cube_scenes_and_task_reset(cubes):
    scene = create_cube_stack(cubes)
    assert isinstance(scene, mujoco.MjSpec)
    simulator = Simulator(scene, robots=[RobotSpec("panda", "panda")])
    task = CubeStackTask(simulator, cubes)
    expert = make_panda_expert(task)
    simulator.target_updater = expert.update
    for index in range(cubes):
        cube = simulator.model.body(f"cube{index}/object_0")
        target = simulator.model.body(f"cube{index}/object_target_0")
        assert (
            simulator.model.jnt_type[simulator.model.joint(f"cube{index}/object_joint_0").id]
            == mujoco.mjtJoint.mjJNT_FREE
        )
        assert simulator.model.body_mocapid[target.id] >= 0
        np.testing.assert_allclose(simulator.data.body(cube.id).xpos[2], 0.82)
        np.testing.assert_allclose(simulator.data.body(target.id).xpos, task.goals[index])
    initial_qpos = simulator.data.qpos.copy()
    initial_mocap = simulator.data.mocap_pos.copy()
    simulator.run_steps(30)
    task.reset()
    expert.reset()
    np.testing.assert_allclose(simulator.data.qpos, initial_qpos)
    np.testing.assert_allclose(simulator.data.mocap_pos, initial_mocap)
    assert expert.stage == 0
    assert not task.status().ogbench_success


def test_two_cubes_are_lifted_and_released_into_supported_stack():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("panda", "panda")])
    task = CubeStackTask(simulator, 2)
    expert = make_panda_expert(task)
    simulator.target_updater = expert.update
    simulator.run_steps(14000)
    status = task.status()
    assert expert.get_stage_name() == "settle"
    assert np.all(task.max_lift > 1.04)
    assert status.ogbench_success
    assert status.released_stable_stack
    assert status.support_contacts
    assert np.all(status.goal_distances < 0.01)
    assert not simulator.data.warning.number.any()


def test_panda_position_targets_are_not_drawn_as_torque_arrows():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("panda", "panda")])
    robot = simulator.robots["panda"]
    scene = mujoco.MjvScene(simulator.model, maxgeom=100)
    draw_joint_torques(scene, robot, simulator.data)
    assert scene.ngeom == 0


def test_task_uses_positions_from_edited_xml_spec():
    scene = create_cube_stack(2)
    scene.body("cube0/object_0").pos = (0.44, -0.2, 0.02)
    scene.body("cube0/object_target_0").pos = (0.44, 0, 0.02)
    simulator = Simulator(scene, robots=[RobotSpec("panda", "panda")])
    task = CubeStackTask(simulator, 2)
    np.testing.assert_allclose(task.starts[0], [0.44, -0.2, 0.82])
    np.testing.assert_allclose(task.goals[0], [0.44, 0, 0.82])


@pytest.mark.parametrize("cubes", [3, 4])
def test_larger_cube_scenes_start_physical_panda_plan(cubes):
    simulator = Simulator(create_cube_stack(cubes), robots=[RobotSpec("panda", "panda")])
    task = CubeStackTask(simulator, cubes)
    expert = make_panda_expert(task)
    simulator.target_updater = expert.update
    simulator.run_steps(200)
    assert expert.get_stage_name().startswith("cube")
    assert np.all(np.isfinite(simulator.data.qpos))
    assert not simulator.data.warning.number.any()
