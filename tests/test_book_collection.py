"""Book collection configuration and physical geometry playback."""

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import BookInsertionExpert, BookTask, create_book_controller
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe
from mujoco_lab.learning.collect_book import Config
from mujoco_lab.learning.config.replay import BookReplayConfig
from mujoco_lab.learning.datasets.episode import load_episode, save_episode
from mujoco_lab.learning.datasets.replay import capture_frame, capture_metadata
from mujoco_lab.learning.envs.book import BookEnv
from mujoco_lab.learning.replay import EpisodeReplay
from mujoco_lab.learning.rollout.collector import collect_episode


def test_book_recording_round_trip_restores_geometry(tmp_path):
    config = Config(max_seconds=0.04, xy_range=0.01, book_yaw_range_degrees=7)
    robot_config = load_robot_config(config.robot)
    simulator = Simulator(
        create_book_insertion(config.book),
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
        dt=config.simulation_dt,
    )
    robot = simulator.robots[config.robot]
    robot.change_controller(create_book_controller(robot, robot_config.controller))
    task = BookTask(simulator)
    env = BookEnv(
        task,
        xy_range=config.xy_range,
        book_yaw_range_degrees=config.book_yaw_range_degrees,
        max_steps=4,
        physics_steps_per_action=5,
    )
    expert = BookInsertionExpert(
        task, recipe=load_recipe(robot.robot_type), planning=config.planning
    )
    metadata = capture_metadata(
        simulator,
        BookReplayConfig(
            book=config.book,
            robot=config.robot,
            physics_steps_per_action=5,
            xy_range=config.xy_range,
            book_yaw_range_degrees=config.book_yaw_range_degrees,
        ),
    )
    geometry = []

    def record():
        geometry.append(simulator.data.geom_xpos.copy())
        return capture_frame(simulator)

    episode = collect_episode(
        env, expert, seed=12, max_steps=4, record_frame=record, replay_metadata=metadata
    )
    path = tmp_path / "book.npz"
    save_episode(path, episode)
    loaded = load_episode(path)
    assert loaded.metadata["replay"]["book"] == "medium"
    assert loaded.metadata["replay"]["xy_range"] == 0.01
    replay = EpisodeReplay(loaded)
    assert replay.frame_dt == pytest.approx(0.01)
    for index in reversed(range(replay.frame_count)):
        replay.set_frame(index)
        np.testing.assert_allclose(replay.simulator.data.geom_xpos, geometry[index], atol=1e-12)
        np.testing.assert_array_equal(replay.simulator.data.qpos, loaded.qpos[index])
