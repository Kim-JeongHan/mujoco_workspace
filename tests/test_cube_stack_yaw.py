"""Heuristic grasp targets for cubes rotated around the tabletop normal."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control import create_controller
from mujoco_lab.planning import default_planning


def _plan_at_yaw(degrees: float, method: str = "heuristic", robot_type: str = "panda"):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec(robot_type, robot_type, config=load_robot_config(robot_type))],
    )
    joint = simulator.model.joint("cube0/object_joint_0")
    qpos = int(joint.qposadr[0])
    simulator.data.qpos[qpos + 3 : qpos + 7] = Rotation.from_euler(
        "z", degrees, degrees=True
    ).as_quat(scalar_first=True)
    mujoco.mj_forward(simulator.model, simulator.data)
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots[robot_type]
    robot.change_controller(
        create_controller(robot, load_robot_config(robot.robot_type).controller)
    )
    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method=method,
        planning=default_planning() if method == "sampling" else None,
    )
    starts = np.array([simulator.data.body(f"cube{i}/object_0").xpos for i in range(2)])
    goals = np.array([simulator.data.body(f"cube{i}/object_target_0").xpos for i in range(2)])
    stages = {stage.name: stage for stage in expert.build_stages(starts, goals)}
    return simulator, expert, starts, goals, stages


def _grasp_pose(simulator, expert, stage):
    data = mujoco.MjData(simulator.model)
    data.qpos[:] = simulator.data.qpos
    data.qpos[expert.robot.state.qpos_indices[:7]] = stage.waypoints[-1]
    mujoco.mj_forward(simulator.model, data)
    site = expert.robot.state.site_id(expert.recipe.frame)
    return data.site_xpos[site].copy(), Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))


@pytest.mark.parametrize("degrees", [-15.0, 15.0, -45.0, 45.0])
@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_pick_tracks_cube_yaw_and_place_returns_to_recipe(degrees, method):
    simulator, expert, starts, goals, stages = _plan_at_yaw(degrees, method)
    yaw = (np.deg2rad(degrees) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    yaw_rotation = Rotation.from_euler("z", yaw)
    if expert.recipe.euler_xyz_degrees is None:
        site = expert.robot.state.site_id(expert.recipe.frame)
        recipe_rotation = Rotation.from_matrix(simulator.data.site_xmat[site].reshape(3, 3))
    else:
        recipe_rotation = Rotation.from_euler("xyz", expert.recipe.euler_xyz_degrees, degrees=True)

    pick_names = (
        ("above_pick", "pick", "close", "lift")
        if method == "heuristic"
        else ("above_pick", "pick", "close")
    )
    for name in pick_names:
        stage = stages[f"cube0:{name}"]
        position, orientation = _grasp_pose(simulator, expert, stage)
        target = starts[0] + yaw_rotation.apply(stage.recipe.offset_xyz_m)
        assert np.linalg.norm(position - target) < 0.009
        assert (yaw_rotation * recipe_rotation * orientation.inv()).magnitude() < 0.34

    place_names = (
        ("above_place", "place", "release", "retract")
        if method == "heuristic"
        else ("place", "release", "retract")
    )
    for name in place_names:
        stage = stages[f"cube0:{name}"]
        position, orientation = _grasp_pose(simulator, expert, stage)
        target = goals[0] + stage.recipe.offset_xyz_m
        assert np.linalg.norm(position - target) < 0.009
        assert (recipe_rotation * orientation.inv()).magnitude() < 0.34


@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_positive_45_degree_tie_uses_negative_45_degree_grasp(method):
    _, _, _, _, positive = _plan_at_yaw(45, method)
    _, _, _, _, negative = _plan_at_yaw(-45, method)
    for name, stage in positive.items():
        np.testing.assert_allclose(negative[name].waypoints[-1], stage.waypoints[-1], atol=1e-9)
