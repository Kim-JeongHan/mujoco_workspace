"""Goal-consistent relabeling, episode boundaries, and fixed-horizon physics."""

from copy import deepcopy

import numpy as np
import pytest

from mujoco_lab.learning.config.replay import BookReplayConfig, CubeStackReplayConfig
from mujoco_lab.learning.envs.factory import create_env
from mujoco_lab.learning.rollout.replay_buffer import HERReplayBuffer, ReplayEpisode


@pytest.mark.parametrize(
    "scene",
    [
        CubeStackReplayConfig(robot="panda", cubes=1),
        CubeStackReplayConfig(robot="panda", cubes=2, environment="warehouse"),
        BookReplayConfig(robot="panda", book="small"),
    ],
)
def test_relabel_updates_all_goal_features_without_changing_physical_observations(scene):
    env = create_env(scene, max_steps=2)
    try:
        obs, _ = env.reset(seed=5)
        before = obs.copy()
        achieved = env.achieved_goal(obs).copy()
        histories = np.broadcast_to(obs, (3, 2, len(obs)))
        goals = np.broadcast_to(achieved, (3, 1, len(achieved))).copy()
        goals[..., :3] += [0.03, 0, 0]
        relabeled = env.relabel_observation(histories, goals)
        np.testing.assert_allclose(
            env.desired_goal(relabeled), np.broadcast_to(goals, (3, 2, len(achieved))), atol=1e-7
        )
        np.testing.assert_array_equal(env.achieved_goal(relabeled), env.achieved_goal(histories))
        np.testing.assert_array_equal(obs, before)
        np.testing.assert_array_equal(env.compute_goal_reward(achieved, achieved), 0)
        assert env.compute_goal_reward(achieved, goals[0, 0]) == -1
        if isinstance(scene, BookReplayConfig):
            rotated = achieved.copy()
            # Negating both columns rotates any starting orientation by 180 degrees.
            rotated[3:] = -achieved[3:]
            assert env.compute_goal_reward(achieved, rotated) == -1
    finally:
        env.close()


class GoalLine:
    """Raw frames encode position, episode identity, and goal-minus-position."""

    def achieved_goal(self, observations):
        return observations[..., :1]

    def desired_goal(self, observations):
        return observations[..., :1] + observations[..., 2:3]

    def relabel_observation(self, observations, goal):
        result = observations.copy()
        result[..., 2:3] = goal - result[..., :1]
        return result

    def compute_goal_reward(self, achieved, desired):
        return np.where(np.abs(achieved - desired).max(axis=-1) < 0.01, 0, -1).astype(np.float32)


def line_episode(identity, steps=3):
    positions = np.arange(steps + 1, dtype=np.float32) + identity * 10
    observations = np.column_stack((positions, np.full_like(positions, identity), 100 - positions))
    return ReplayEpisode(
        observations,
        np.arange(steps, dtype=np.float32)[:, None] / steps,
        np.zeros(steps, dtype=bool),
        np.arange(steps) == steps - 1,
    )


def test_future_goals_and_padded_histories_stay_within_the_sampled_episode():
    np.random.seed(13)
    buffer = HERReplayBuffer(10)
    buffer.add(line_episode(0))
    buffer.add(line_episode(1))
    saved = deepcopy(buffer.episodes)
    sample = buffer.sample(256, GoalLine(), obs_horizon=4, her_ratio=1)
    goals = GoalLine().desired_goal(sample.states)
    current = sample.states[:, -1, 0]
    identity = sample.states[:, -1, 1]
    assert np.all(goals[:, -1, 0] > current)
    assert np.all(goals[:, -1, 0] <= identity * 10 + 3)
    np.testing.assert_array_equal(goals, np.broadcast_to(goals[:, -1:], goals.shape))
    np.testing.assert_array_equal(
        sample.states[..., 1], np.broadcast_to(identity[:, None], (256, 4))
    )
    np.testing.assert_array_equal(sample.next_states[:, -1, 0], current + 1)
    expected_rewards = np.where(sample.next_states[:, -1, 0] == goals[:, -1, 0], 0, -1)
    np.testing.assert_array_equal(sample.rewards, expected_rewards)
    np.testing.assert_array_equal(
        sample.next_states[sample.truncated, -1, 0], identity[sample.truncated] * 10 + 3
    )
    assert sample.relabeled.all() and not sample.terminated.any()
    for actual, original in zip(buffer.episodes, saved, strict=True):
        np.testing.assert_array_equal(actual.observations, original.observations)


def test_original_goal_replay_and_eviction_retain_complete_episodes():
    buffer = HERReplayBuffer(5)
    buffer.add(line_episode(0, 3))
    buffer.add(line_episode(1, 2))
    buffer.add(line_episode(2, 2))
    assert len(buffer) == 4
    sample = buffer.sample(32, GoalLine(), obs_horizon=2, her_ratio=0)
    assert set(sample.states[:, -1, 1]) == {1, 2}
    np.testing.assert_array_equal(GoalLine().desired_goal(sample.states), 100)
    assert not sample.relabeled.any()
    np.testing.assert_array_equal(sample.rewards, -1)


def test_fixed_horizon_still_measures_success_on_every_physics_tick(monkeypatch):
    env = create_env(
        CubeStackReplayConfig(robot="panda", cubes=1, physics_steps_per_action=3),
        max_steps=1,
        terminate_on_success=False,
    )
    try:
        env.reset(seed=1)
        calls = []

        def success():
            calls.append(env.data.time)
            return True

        monkeypatch.setattr(env, "_is_success", success)
        _, reward, terminated, truncated, info = env.step(
            (env.action_space.low + env.action_space.high) / 2
        )
        assert len(calls) == 3
        assert info["physics_steps"] == 3 and info["success"]
        assert (reward, terminated, truncated) == (1, False, True)
    finally:
        env.close()
