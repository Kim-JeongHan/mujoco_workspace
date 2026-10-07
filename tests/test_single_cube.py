"""Single-cube scene, collection, replay, and training selection."""

import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.datasets.replay import capture_frame, capture_metadata
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.replay import EpisodeReplay
from mujoco_lab.learning.rollout import collect_episode


@pytest.mark.parametrize("robot_name", ["panda", "forte"])
def test_single_cube_env_recording_replays_with_39_observations(robot_name):
    simulator = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec(robot_name, robot_name, config=load_robot_config(robot_name))],
    )
    robot = simulator.robots[robot_name]
    controller = (
        create_test_controller(robot, controller="position", frame="grasp")
        if robot_name == "panda"
        else create_test_controller(robot, controller="pd", frame="grasp")
    )
    robot.change_controller(controller)
    task = CubeStackTask(simulator, 1)
    env = CubeStackEnv(task, xy_range=0, cube_yaw_range_degrees=0, physics_steps_per_action=5)
    initial_42, _ = env.reset(seed=42)
    initial_43, _ = env.reset(seed=43)
    np.testing.assert_array_equal(initial_42, initial_43)
    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method="heuristic",
    )
    metadata = capture_metadata(
        simulator, CubeStackReplayConfig(cubes=1, robot=robot_name, physics_steps_per_action=5)
    )
    evaluation, scene = create_evaluation_env(
        {"dataset_metadata": {"replay": metadata}},
        xy_range=0,
        min_gap=0.01,
        max_steps=2,
        cube_yaw_range_degrees=0,
    )
    assert evaluation.observation_space.shape == (39,)
    assert scene["cubes"] == 1
    episode = collect_episode(
        env,
        expert,
        seed=42,
        max_steps=2,
        record_frame=lambda: capture_frame(simulator),
        replay_metadata=metadata,
    )
    assert episode.states.shape == (3, 39)
    assert episode.actions.shape == (2, 8)
    assert episode.metadata["replay"]["scene"] == "cube_stack"
    assert episode.metadata["replay"]["robot"] == robot_name
    assert episode.metadata["replay"]["cubes"] == 1
    replay = EpisodeReplay(episode)
    replay.set_frame(2)
    np.testing.assert_array_equal(replay.simulator.data.qpos, episode.qpos[-1])
