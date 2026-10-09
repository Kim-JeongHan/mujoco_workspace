"""Collect completed motion and failures without synthesizing expert actions."""

import gymnasium as gym
import numpy as np
import pytest

from mujoco_lab.behaviors import Expert
from mujoco_lab.learning.rollout import collect_episode


class CountingEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,))

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.received = []
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.received.append(action.copy())
        # Environments may mutate their action argument.
        action.fill(-1)
        success = self.steps == 3
        return (
            np.full(2, self.steps, dtype=np.float32),
            float(success),
            success,
            False,
            {"success": success},
        )


class EndingExpert(Expert):
    def __init__(self, action_count, *, fail):
        super().__init__()
        self.action_count = action_count
        self.fail = fail

    def reset(self):
        self.calls = 0
        self.failed = False
        self.failure_reason = None

    def act(self, *, dt=0.0):
        self.calls += 1
        if self.calls > self.action_count:
            self.failed = self.fail
            self.failure_reason = "no route" if self.fail else None
            return None
        return np.full(8, 0.25, dtype=np.float32)


def test_completed_motion_retains_last_action_until_environment_success():
    env = CountingEnv()
    episode = collect_episode(env, EndingExpert(1, fail=False), max_steps=10)

    assert len(episode) == env.steps == 3
    np.testing.assert_array_equal(episode.actions, np.full((3, 8), 0.25))
    np.testing.assert_array_equal(env.received, episode.actions)
    assert episode.metadata["success"] is True
    assert episode.metadata["expert_failed"] is False


def test_initial_completion_records_an_empty_episode_without_stepping():
    env = CountingEnv()
    episode = collect_episode(env, EndingExpert(0, fail=False), max_steps=10)

    assert len(episode) == env.steps == 0
    assert episode.actions.shape == (0, 8)
    assert episode.states.shape == (1, 2)
    assert episode.metadata["termination_reason"] == "expert_complete"
    assert episode.metadata["expert_failed"] is False
    assert episode.metadata["success"] is None


@pytest.mark.parametrize("action_count", [0, 1])
def test_failure_without_action_stops_without_an_extra_environment_step(action_count):
    env = CountingEnv()
    episode = collect_episode(env, EndingExpert(action_count, fail=True), max_steps=10)

    assert len(episode) == env.steps == action_count
    assert episode.actions.shape == (action_count, 8)
    assert episode.states.shape == (action_count + 1, 2)
    assert episode.metadata["expert_failed"] is True
    assert episode.metadata["termination_reason"] == "no route"
    assert episode.metadata["collector_timeout"] is False
    if action_count:
        assert episode.truncated is not None
        assert episode.truncated[-1]
