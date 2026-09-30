"""Policy cadence and replay compatibility at the training/evaluation boundary."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation import evaluate_policy
from mujoco_lab.learning.policies.factory import build_policy


def _stats():
    return Normalizer(
        state_mean=np.zeros(1, dtype=np.float32),
        state_std=np.ones(1, dtype=np.float32),
        action_mean=np.zeros(1, dtype=np.float32),
        action_std=np.ones(1, dtype=np.float32),
    )


class CadenceEnv:
    physics_steps_per_action = 5
    task = None

    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = SimpleNamespace(
            shape=(1,), low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )
        self.simulator = SimpleNamespace(dt=0.002, data=SimpleNamespace(time=0.0))
        self.steps = 0

    def reset(self, *, seed):
        self.steps = 0
        self.simulator.data.time = 0.0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.simulator.data.time += 0.004 if self.steps == 5 else 0.01
        done = self.steps == 5
        return (
            np.array([float(self.steps)], dtype=np.float32),
            0.0,
            done,
            False,
            {
                "success": done,
                "termination_reason": "success" if done else None,
            },
        )


def test_chunk16_executes_four_at_100hz_and_stops_inside_next_chunk():
    model = build_policy("mse", state_dim=2, action_dim=1, chunk_size=16, hidden_dims=())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    seen = []
    handle = model.net[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().cpu().numpy().copy())
    )
    metadata = {
        "architecture": {
            "frame_dim": 1,
            "state_dim": 2,
            "action_dim": 1,
            "obs_horizon": 2,
            "chunk_size": 16,
            "execution_horizon": 4,
            "physics_steps_per_action": 5,
        },
        "train_config": {"physics_steps_per_action": 5},
        "dataset_metadata": {"replay": {"physics_steps_per_action": 5}},
    }
    env = CadenceEnv()
    rows, _ = evaluate_policy(
        cast(CubeStackEnv, env),
        model,
        _stats(),
        metadata,
        num_episodes=1,
        seed=1,
        policy_seed=2,
        max_steps=30,
        device=torch.device("cpu"),
        flow_num_steps=1,
    )
    handle.remove()
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], [[0.0, 0.0]])
    np.testing.assert_array_equal(seen[1], [[3.0, 4.0]])
    assert rows[0]["steps"] == 5
    assert rows[0]["sim_seconds"] == pytest.approx(0.044)
    assert rows[0]["success"]


def test_episode_without_action_cadence_is_rejected():
    episode = Episode(
        states=np.zeros((3, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        metadata={"success": True, "replay": {"dt": 0.002}},
    )
    with pytest.raises(ValueError, match="physics_steps_per_action must be a positive integer"):
        episode.check_physics_step_consistency(5)
    compatible = Episode(
        states=episode.states,
        actions=episode.actions,
        metadata={"replay": {"dt": 0.002, "physics_steps_per_action": 5}},
    )
    compatible.check_physics_step_consistency(5)
    with pytest.raises(ValueError, match="physics_steps_per_action=5 differs from config 1"):
        compatible.check_physics_step_consistency(1)
