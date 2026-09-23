"""Policy cadence and replay compatibility at the training/evaluation boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.loading import validate_episode_cadence
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation import evaluate_policy
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.trainers.train_bc import run_training


def _stats():
    return Normalizer(
        state_mean=np.zeros(1, dtype=np.float32),
        state_std=np.ones(1, dtype=np.float32),
        action_mean=np.zeros(1, dtype=np.float32),
        action_std=np.ones(1, dtype=np.float32),
    )


class CadenceEnv:
    physics_steps_per_action = 5

    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = SimpleNamespace(
            shape=(1,), low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )
        self.simulator = SimpleNamespace(data=SimpleNamespace(time=0.0))
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
        env,
        model,
        _stats(),
        metadata,
        num_episodes=1,
        seed=1,
        policy_seed=2,
        max_steps=30,
        dt=0.002,
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


def test_legacy_500hz_episode_rejected_by_100hz_config():
    episode = Episode(
        states=np.zeros((3, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        metadata={"success": True, "replay": {"dt": 0.002}},
    )
    with pytest.raises(ValueError, match="Episode 0 physics_steps_per_action=1"):
        validate_episode_cadence([episode], 5)
    config = TrainConfig(num_epochs=1)
    with pytest.raises(ValueError, match="Episode 0 physics_steps_per_action=1"):
        run_training(config, [episode])
    compatible = Episode(
        states=episode.states,
        actions=episode.actions,
        metadata={"replay": {"dt": 0.002, "physics_steps_per_action": 5}},
    )
    with pytest.raises(ValueError, match="Episode 1 physics_steps_per_action=1"):
        run_training(config, [compatible], [episode])


def test_training_rejects_mixed_physics_dt():
    episodes = [
        Episode(
            states=np.zeros((2, 1), dtype=np.float32),
            actions=np.zeros((1, 1), dtype=np.float32),
            metadata={"replay": {"dt": dt, "physics_steps_per_action": 5}},
        )
        for dt in (0.002, 0.004)
    ]
    with pytest.raises(ValueError, match="physics dt differs"):
        validate_episode_cadence(episodes, 5)


def test_legacy_checkpoint_without_repeat_restores_one(tmp_path):
    config = TrainConfig(
        obs_horizon=1, chunk_size=1, execution_horizon=1, physics_steps_per_action=1, hidden_dims=()
    )
    model = build_policy("mse", state_dim=1, action_dim=1, chunk_size=1, hidden_dims=())
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, _stats(), config, optimizer_step=1)
    payload = torch.load(path, weights_only=True)
    del payload["architecture"]["physics_steps_per_action"]
    del payload["train_config"]["physics_steps_per_action"]
    torch.save(payload, path)
    _, _, metadata = load_checkpoint(path)
    assert metadata["architecture"]["physics_steps_per_action"] == 1
    assert metadata["train_config"]["physics_steps_per_action"] == 1
