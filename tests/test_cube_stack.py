"""OGBench-layout cubes and a real Panda grasp, lift, and released stack."""

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import Constraints, ControllerConfig, PoseConfig, RobotConfig
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe


def make_panda_expert(task):
    robot = task.simulator.robots["panda"]
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    return CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method="heuristic",
    )


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
    assert task.status().is_final_success
    # Exact goal coordinates do not yet establish contact between the two cubes.
    assert not task.status().is_released_stable
    assert simulator.data.time == 0
    np.testing.assert_array_equal(task.completed_placements(), [True, False])

    for _ in range(400):
        task.status()
        simulator.run_steps(1)
    assert updates
    assert task.status().is_released_stable
    assert task.status().is_support_contacts

    joint = simulator.model.joint("cube1/object_joint_0")
    simulator.data.qpos[int(joint.qposadr[0])] += 0.031
    mujoco.mj_forward(simulator.model, simulator.data)
    assert task.status().is_final_success
    assert not task.status().is_released_stable
    simulator.data.qpos[int(joint.qposadr[0])] -= 0.031
    mujoco.mj_forward(simulator.model, simulator.data)
    dof = int(joint.dofadr[0])
    simulator.data.qvel[dof : dof + 6] = [0.1, 0, 0, 0, 0, 1]
    assert task.status().is_released_stable
    simulator.data.qvel[dof : dof + 6] = 0
    simulator.data.qpos[int(joint.qposadr[0]) + 2] += 0.1
    mujoco.mj_forward(simulator.model, simulator.data)
    assert not task.status().is_support_contacts
    assert not task.status().is_released_stable
    before_qpos = simulator.data.qpos.copy()
    before_time = simulator.data.time
    task.reset()
    np.testing.assert_array_equal(simulator.data.qpos, before_qpos)
    assert simulator.data.time == before_time
    assert not task.completed_placements().any()
    assert simulator.target_updater is updater
    assert not task.status().is_final_success


@pytest.mark.parametrize(
    ("xy_offset", "height_error", "success"),
    [(0.025, 0.0, True), (0.031, 0.0, False), (0.0, 0.011, True), (0.0, 0.013, False)],
)
def test_supported_cube_placement_uses_xy_and_height_tolerances(xy_offset, height_error, success):
    scene = create_cube_stack(1)
    scene.body("cube0/object_0").pos = (0.425 + xy_offset, 0, 0.02)
    scene.body("cube0/object_target_0").pos = (0.425, 0, 0.02 + height_error)
    simulator = Simulator(scene)
    task = CubeStackTask(simulator, 1)

    status = task.status()

    assert status.is_support_contacts
    assert status.is_released_stable == success
    assert simulator.data.time == 0


def test_task_rejects_robot_contact_at_goal(tmp_path, monkeypatch):
    path = tmp_path / "blocker.xml"
    path.write_text("""
        <mujoco><worldbody><body name="base">
          <joint name="anchor" type="slide" axis="1 0 0" limited="true" range="-.001 .001"/>
          <geom name="touch" type="sphere" pos=".425 0 2" size=".02"/>
        </body></worldbody>
        <actuator><position name="hold" joint="anchor" kp="1000"/></actuator></mujoco>
    """)
    monkeypatch.setitem(ROBOT_ASSETS, "blocker", RobotAsset(path))
    simulator = Simulator(
        create_cube_stack(2),
        robots=[
            RobotSpec(
                "blocker",
                "blocker",
                config=RobotConfig(
                    ControllerConfig("position"),
                    Constraints(["anchor"], [[-0.001, 0.001]], [0.01], [0.1]),
                    pose=PoseConfig([0]),
                ),
            )
        ],
    )
    task = CubeStackTask(simulator, 2)
    place_cubes_at_goals(simulator, task)
    for _ in range(400):
        task.status()
        simulator.run_steps(1)
    assert task.status().is_released_stable

    simulator.model.geom("blocker/touch").pos[:] = [0.425, 0, 0.02]
    mujoco.mj_forward(simulator.model, simulator.data)
    assert any(
        "blocker/touch"
        in (simulator.model.geom(contact.geom1).name, simulator.model.geom(contact.geom2).name)
        for contact in simulator.data.contact
    )
    assert task.status().is_final_success
    assert not task.status().is_released_stable


def test_two_cubes_are_lifted_and_released_into_supported_stack():
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_panda_expert(task)
    simulator.target_updater = expert.update
    simulator.run_steps(14000)
    status = task.status()
    assert expert.get_stage_name() == "settle"
    assert np.all(task.max_height > 1.04)
    assert status.is_final_success
    assert status.is_released_stable
    assert status.is_support_contacts
    assert np.all(status.position_errors < 0.01)
    assert not simulator.data.warning.number.any()


def test_task_uses_positions_from_edited_xml_spec():
    scene = create_cube_stack(2)
    scene.body("cube0/object_0").pos = (0.44, -0.2, 0.02)
    scene.body("cube0/object_target_0").pos = (0.44, 0, 0.02)
    simulator = Simulator(
        scene, robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))]
    )
    task = CubeStackTask(simulator, 2)
    np.testing.assert_allclose(task.starts[0], [0.44, -0.2, 0.82])
    np.testing.assert_allclose(task.goals[0], [0.44, 0, 0.82])
