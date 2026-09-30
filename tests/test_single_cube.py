"""Single-cube scene, collection, replay, and training selection."""

import sys

import mujoco
import numpy as np
import pytest
import tyro

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.control import create_controller
from mujoco_lab.learning.collect import Config as CollectConfig
from mujoco_lab.learning.collect import create_expert
from mujoco_lab.learning.datasets.episode import Episode, save_episode
from mujoco_lab.learning.datasets.replay import capture_frame, cube_stack_metadata
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.replay import EpisodeReplay
from mujoco_lab.learning.rollout import collect_episode
from mujoco_lab.tasks import CubeStackTask, default_planning


@pytest.mark.parametrize("environment", ["table_shelf", "warehouse"])
def test_single_cube_scene_has_original_first_cube_and_table_goal(environment):
    simulator = Simulator(create_cube_stack(1, environment=environment))
    task = CubeStackTask(simulator, 1)
    np.testing.assert_allclose(task.starts, [[0.425, -0.2, 0.82]])
    np.testing.assert_allclose(task.goals, [[0.425, 0, 0.82]])
    assert simulator.model.nmocap == 1
    assert mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, "cube1/object_0") == -1
    has_small_shelf = (
        mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_GEOM, "small_shelf/box") >= 0
    )
    assert has_small_shelf == (environment == "warehouse")


@pytest.mark.parametrize("robot_name", ["panda", "forte"])
def test_single_cube_env_recording_replays_with_39_observations(robot_name):
    simulator = Simulator(create_cube_stack(1), robots=[RobotSpec(robot_name, robot_name)])
    robot = simulator.robots[robot_name]
    controller = (
        create_controller("position", robot, gravity_compensation=True, frame="grasp")
        if robot_name == "panda"
        else create_controller("pd", robot, frame="grasp")
    )
    robot.change_controller(controller)
    task = CubeStackTask(simulator, 1)
    env = CubeStackEnv(task, xy_range=0, cube_yaw_range_degrees=0, physics_steps_per_action=5)
    initial_42, _ = env.reset(seed=42)
    initial_43, _ = env.reset(seed=43)
    np.testing.assert_array_equal(initial_42, initial_43)
    expert = create_expert(task, planning=default_planning())
    metadata = cube_stack_metadata(simulator, cubes=1, robot=robot_name, physics_steps_per_action=5)
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


def test_collection_cli_accepts_both_bundled_robots():
    for robot in ("panda", "forte"):
        assert tyro.cli(CollectConfig, args=["--robot", robot]).robot == robot


@pytest.mark.parametrize("cubes", [1, 2, 3, 4])
def test_collection_cli_accepts_all_bundled_cube_counts(cubes):
    assert tyro.cli(CollectConfig, args=["--cubes", str(cubes)]).cubes == cubes


@pytest.mark.parametrize(("robot_name", "cubes"), [("panda", 1), ("panda", 2), ("forte", 1)])
def test_expert_releases_stably_at_five_physics_ticks_per_action(robot_name, cubes):
    simulator = Simulator(create_cube_stack(cubes), robots=[RobotSpec(robot_name, robot_name)])
    robot = simulator.robots[robot_name]
    controller = (
        create_controller("position", robot, gravity_compensation=True, frame="grasp")
        if robot_name == "panda"
        else create_controller("pd", robot, frame="grasp")
    )
    robot.change_controller(controller)
    task = CubeStackTask(simulator, cubes)
    env = CubeStackEnv(
        task,
        xy_range=0,
        cube_yaw_range_degrees=0,
        physics_steps_per_action=5,
        max_steps=6000,
    )
    expert = create_expert(task, planning=default_planning())
    episode = collect_episode(env, expert, seed=42, max_steps=6000)
    assert episode.metadata["success"] is True
    assert episode.terminated[-1]
    assert task.max_lift[0] > task.starts[0, 2] + 0.04
    assert task.status().released_stable_stack
    assert task.status().support_contacts
    assert not simulator.data.warning.number.any()


def test_training_uses_single_cube_count_from_recording(tmp_path, monkeypatch):
    from mujoco_lab.learning.train import main

    episode = Episode(
        states=np.zeros((2, 39), dtype=np.float32),
        actions=np.zeros((1, 8), dtype=np.float32),
        metadata={
            "success": True,
            "replay": {
                "scene": "cube_stack",
                "cubes": 1,
                "dt": 0.002,
                "physics_steps_per_action": 5,
            },
        },
    )
    save_episode(tmp_path / "episode_000000.npz", episode)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--eval-interval",
            "0",
            "--validation-ratio",
            "0",
        ],
    )

    class SplitReached(Exception):
        pass

    def stop_after_split(_config, train, validation, **_kwargs):
        assert len(train) == 1 and not validation
        assert train[0].metadata["replay"]["cubes"] == 1
        raise SplitReached

    monkeypatch.setattr("mujoco_lab.learning.train.run_training", stop_after_split)
    with pytest.raises(SplitReached):
        main()
