"""Heuristic grasp targets for cubes rotated around the tabletop normal."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.planning import planner_from_config
from mujoco_lab.tasks import (
    BaseCubeStackMotionGenerator,
    CubeStackTask,
    HeuristicCubeStackMotionGenerator,
    SamplingCubeStackMotionGenerator,
    default_planning,
)


def _plan_at_yaw(degrees: float, method: str = "heuristic", robot_type: str = "panda"):
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec(robot_type, robot_type)])
    joint = simulator.model.joint("cube0/object_joint_0")
    qpos = int(joint.qposadr[0])
    simulator.data.qpos[qpos + 3 : qpos + 7] = Rotation.from_euler(
        "z", degrees, degrees=True
    ).as_quat(scalar_first=True)
    mujoco.mj_forward(simulator.model, simulator.data)
    task = CubeStackTask(simulator, 2)
    generator = (
        SamplingCubeStackMotionGenerator(task, planner=planner_from_config(default_planning()))
        if method == "sampling"
        else HeuristicCubeStackMotionGenerator(task)
    )
    starts = np.array([simulator.data.body(f"cube{i}/object_0").xpos for i in range(2)])
    goals = np.array([simulator.data.body(f"cube{i}/object_target_0").xpos for i in range(2)])
    stages = {
        stage.name: stage
        for stage in BaseCubeStackMotionGenerator.generate(generator, starts, goals)
    }
    return simulator, generator, starts, goals, stages


def _grasp_pose(simulator, generator, stage):
    data = mujoco.MjData(simulator.model)
    data.qpos[:] = simulator.data.qpos
    data.qpos[generator.robot.state.qpos_indices[:7]] = stage.command[:7]
    mujoco.mj_forward(simulator.model, data)
    site = generator.robot.state.site_id(generator.recipe.frame)
    return data.site_xpos[site].copy(), Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))


@pytest.mark.parametrize("degrees", [-15.0, 15.0, -45.0, 45.0])
@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_pick_tracks_cube_yaw_and_place_returns_to_recipe(degrees, method):
    simulator, generator, starts, goals, stages = _plan_at_yaw(degrees, method)
    yaw = (np.deg2rad(degrees) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    yaw_rotation = Rotation.from_euler("z", yaw)
    if generator.recipe.euler_xyz_degrees is None:
        site = generator.robot.state.site_id(generator.recipe.frame)
        recipe_rotation = Rotation.from_matrix(simulator.data.site_xmat[site].reshape(3, 3))
    else:
        recipe_rotation = Rotation.from_euler(
            "xyz", generator.recipe.euler_xyz_degrees, degrees=True
        )

    pick_names = (
        ("above_pick", "pick", "close", "lift") if method == "heuristic" else ("pick", "close")
    )
    for name in pick_names:
        stage = stages[f"cube0:{name}"]
        position, orientation = _grasp_pose(simulator, generator, stage)
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
        position, orientation = _grasp_pose(simulator, generator, stage)
        target = goals[0] + stage.recipe.offset_xyz_m
        assert np.linalg.norm(position - target) < 0.009
        assert (recipe_rotation * orientation.inv()).magnitude() < 0.34


@pytest.mark.parametrize("degrees", [90.0, -90.0, 180.0, 270.0])
@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_quarter_turns_use_the_same_targets_as_zero_yaw(degrees, method):
    _, _, _, _, baseline = _plan_at_yaw(0, method)
    _, _, _, _, rotated = _plan_at_yaw(degrees, method)
    for name, stage in baseline.items():
        np.testing.assert_allclose(rotated[name].command, stage.command, atol=1e-9)


@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_positive_45_degree_tie_uses_negative_45_degree_grasp(method):
    _, _, _, _, positive = _plan_at_yaw(45, method)
    _, _, _, _, negative = _plan_at_yaw(-45, method)
    for name, stage in positive.items():
        np.testing.assert_allclose(negative[name].command, stage.command, atol=1e-9)


def test_forte_sampling_pick_target_tracks_rotated_cube():
    simulator, generator, starts, goals, stages = _plan_at_yaw(15, "sampling", "forte")
    pick = stages["cube0:pick"]
    place = stages["cube0:place"]
    pick_position, pick_orientation = _grasp_pose(simulator, generator, pick)
    place_position, place_orientation = _grasp_pose(simulator, generator, place)
    nominal = Rotation.from_euler("xyz", generator.recipe.euler_xyz_degrees, degrees=True)
    yaw = Rotation.from_euler("z", 15, degrees=True)
    assert np.linalg.norm(pick_position - (starts[0] + yaw.apply(pick.recipe.offset_xyz_m))) < 0.009
    assert (yaw * nominal * pick_orientation.inv()).magnitude() < 0.34
    assert np.linalg.norm(place_position - (goals[0] + place.recipe.offset_xyz_m)) < 0.009
    assert (nominal * place_orientation.inv()).magnitude() < 0.34
