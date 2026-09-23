"""Physical action resampling preserves reset conditions and the new timebase."""

from dataclasses import replace

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.control import create_controller
from mujoco_lab.learning.datasets.episode import Episode, load_episode, save_episode
from mujoco_lab.learning.datasets.replay import cube_stack_metadata
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.resample import _source_replay, replay_episode
from mujoco_lab.tasks import CubeStackTask


@pytest.fixture
def short_source(tmp_path):
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    replay = cube_stack_metadata(simulator, cubes=2, robot="forte")
    robot = simulator.robots["forte"]
    robot.change_controller(create_controller("pd", robot, frame="grasp"))
    env = CubeStackEnv(CubeStackTask(simulator, 2), max_steps=10)
    observation, _ = env.reset(seed=7)
    states = [observation.copy()]
    actions = []
    times = [float(simulator.data.time)]
    action = np.clip(
        np.r_[observation[:7], observation[14]], env.action_space.low, env.action_space.high
    ).astype(np.float32)
    for _ in range(5):
        observation, _, _, _, _ = env.step(action)
        states.append(observation.copy())
        actions.append(action.copy())
        times.append(float(simulator.data.time))
    source = Episode(
        states=np.stack(states),
        actions=np.stack(actions),
        frame_times=np.asarray(times),
        metadata={"seed": 7, "success": False, "replay": replay},
    )
    path = tmp_path / "episode_000000.npz"
    save_episode(path, source)
    return path, load_episode(path)


def test_short_physical_replay_records_new_endpoints_and_final_full_hold(short_source):
    path, source = short_source
    replay = {**source.metadata["replay"], "physics_steps_per_action": 2}
    env, _, _ = create_evaluation_env(
        {"dataset_metadata": {"replay": replay}},
        xy_range=0.02,
        min_gap=0.01,
        max_steps=4,
    )
    result = replay_episode(
        env, source, source_path=path, physics_steps_per_action=2, replay_metadata=replay
    )
    assert result.actions.shape[0] == 3
    assert result.states.shape[0] == result.qpos.shape[0] == 4
    assert result.metadata["source_action_indices"] == [0, 2, 4]
    assert result.metadata["termination_reason"] == "source_exhausted"
    assert result.metadata["success"] is False
    assert result.metadata["final_action_appended"] is False
    assert result.metadata["final_action_executed"] is True
    assert result.truncated.tolist() == [False, False, True]
    np.testing.assert_allclose(result.frame_times, [0, 0.004, 0.008, 0.012], atol=1e-10)
    np.testing.assert_array_equal(result.actions, source.actions[[0, 2, 4]])
    with pytest.raises(ValueError, match="action repeat"):
        replay_episode(
            env,
            source,
            source_path=path,
            physics_steps_per_action=2,
            replay_metadata=source.metadata["replay"],
        )


def test_missing_final_source_action_is_executed_once(short_source):
    path, source = short_source
    final_action = source.actions[-1].copy()
    final_action[0] += 0.001
    source = replace(
        source,
        states=np.concatenate([source.states, source.states[-1:]]),
        actions=np.concatenate([source.actions, final_action[None]]),
        frame_times=np.arange(7, dtype=np.float64) * 0.002,
    )
    replay = {**source.metadata["replay"], "physics_steps_per_action": 2}
    env, _, _ = create_evaluation_env(
        {"dataset_metadata": {"replay": replay}},
        xy_range=0.02,
        min_gap=0.01,
        max_steps=4,
    )
    result = replay_episode(
        env,
        source,
        source_path=path,
        physics_steps_per_action=2,
        replay_metadata=replay,
    )
    assert result.metadata["source_action_indices"] == [0, 2, 4, 5]
    assert result.metadata["final_action_appended"] is True
    assert result.metadata["final_action_executed"] is True
    assert result.states.shape[0] == result.qpos.shape[0] == 5
    assert result.truncated.tolist() == [False, False, False, True]
    np.testing.assert_allclose(result.frame_times, [0, 0.004, 0.008, 0.012, 0.016], atol=1e-10)
    np.testing.assert_array_equal(result.actions, source.actions[[0, 2, 4, 5]])
    np.testing.assert_array_equal(result.actions[-1], final_action)

    env.max_steps = 1
    result = replay_episode(
        env, source, source_path=path, physics_steps_per_action=2, replay_metadata=replay
    )
    assert result.metadata["source_action_indices"] == [0]
    assert result.metadata["final_action_appended"] is True
    assert result.metadata["final_action_executed"] is False
    assert len(result.actions) == 1


def test_repeat_one_and_single_action_do_not_duplicate_final_source(short_source):
    path, source = short_source
    replay = {**source.metadata["replay"], "physics_steps_per_action": 1}
    env, _, _ = create_evaluation_env(
        {"dataset_metadata": {"replay": replay}}, xy_range=0.02, min_gap=0.01, max_steps=7
    )
    result = replay_episode(
        env, source, source_path=path, physics_steps_per_action=1, replay_metadata=replay
    )
    assert result.metadata["source_action_indices"] == [0, 1, 2, 3, 4]
    assert result.metadata["final_action_appended"] is False
    assert result.metadata["final_action_executed"] is True
    one_action = replace(
        source,
        states=source.states[:2],
        actions=source.actions[:1],
        frame_times=source.frame_times[:2],
    )
    result = replay_episode(
        env, one_action, source_path=path, physics_steps_per_action=1, replay_metadata=replay
    )
    assert result.metadata["source_action_indices"] == [0]


def test_source_requires_seed_and_physical_timebase(short_source):
    _, source = short_source
    with pytest.raises(ValueError, match="recorded seed"):
        _source_replay(replace(source, metadata={**source.metadata, "seed": None}))
    with pytest.raises(ValueError, match="one physics step"):
        _source_replay(
            replace(
                source,
                metadata={
                    **source.metadata,
                    "replay": {**source.metadata["replay"], "physics_steps_per_action": 5},
                },
            )
        )
    path, _ = short_source
