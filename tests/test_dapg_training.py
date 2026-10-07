"""BC initialization, demonstration loading, full training, and exact boundary resume."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from mujoco_lab.learning.algorithms.dapg import Demonstrations
from mujoco_lab.learning.algorithms.npg import NPGConfig
from mujoco_lab.learning.checkpoint import save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.config.dapg import DAPGConfig
from mujoco_lab.learning.datasets.episode import Episode, save_episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.rl_checkpoint import (
    TrainingState,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
)
from mujoco_lab.learning.trainers import train_dapg


class BanditEnv:
    """A one-step quadratic task with a known optimal action of 0.5."""

    def __init__(self):
        self.action_space = spaces.Box(-2, 2, (1,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed=None):
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        error = float(action[0]) - 0.5
        return (
            np.array([1.0], dtype=np.float32),
            -(error**2),
            True,
            False,
            {"success": abs(error) < 0.2},
        )

    def close(self):
        self.closed = True


def create_bc_data(tmp_path):
    config = TrainConfig(
        hidden_dims=(8,),
        obs_horizon=2,
        chunk_size=3,
        execution_horizon=2,
        action_execution_hz=500,
    )
    model = MSEPolicy(2, 1, 3, hidden_dims=(8,))
    normalizer = Normalizer(
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    replay = {
        "scene": "cube_stack",
        "robot": "panda",
        "robot_name": "panda",
        "environment": "table_shelf",
        "cubes": 1,
        "dt": 0.002,
        "physics_steps_per_action": 1,
        "gripper_action_units": "opening_width_m",
        "cube_yaw_range_degrees": 0.0,
    }
    observation = {"frame_dim": 1, "rotation_indices": []}
    dataset_metadata = {"replay": replay, "observation": observation, "train_seeds": [0]}
    path = tmp_path / "bc.pt"
    save_checkpoint(
        path, model, normalizer, config, optimizer_step=12, dataset_metadata=dataset_metadata
    )
    data_dir = tmp_path / "demos"
    data_dir.mkdir()
    for seed in range(2):
        save_episode(
            data_dir / f"episode_{seed:06d}.npz",
            Episode(
                states=np.ones((5, 1), dtype=np.float32),
                actions=np.full((4, 1), 0.5, dtype=np.float32),
                metadata={
                    "success": True,
                    "seed": seed,
                    "replay": replay,
                    "observation": observation,
                },
            ),
        )
    return model, DAPGConfig(
        init_from=path,
        data_dir=data_dir,
        output_dir=tmp_path / "log",
        total_steps=16,
        rollout_steps=8,
        critic_epochs=1,
        critic_batch_size=4,
        critic_n_layers=1,
        critic_layer_size=8,
        eval_interval=1,
        eval_episodes=1,
        checkpoint_interval=1,
        npg=NPGConfig(cg_iterations=4),
    )


def test_initialization_keeps_first_bc_action_and_observation_history(tmp_path):
    bc, config = create_bc_data(tmp_path)
    state = train_dapg.initialize_training(config)
    observations = torch.randn(5, 2)
    # Slicing the output head changes GEMM width, allowing float32 roundoff.
    torch.testing.assert_close(state.actor(observations).mean, bc(observations)[:, :1])
    for name, tensor in state.actor.mean_head.state_dict().items():
        torch.testing.assert_close(tensor, bc.net[-1].state_dict()[name][:1], rtol=0, atol=0)
    assert state.actor.chunk_size == 1
    assert state.metadata["architecture"]["execution_horizon"] == 1
    assert state.metadata["source_bc_architecture"]["chunk_size"] == 3
    assert state.metadata["architecture"]["obs_horizon"] == 2
    demos = train_dapg.load_demonstrations(state)
    # The held-out seed 1 must not become demonstration supervision.
    assert demos.states.shape == (4, 2)
    assert demos.actions.shape == (4, 1, 1)


def test_checkpoint_resume_reproduces_next_training_iteration(tmp_path):
    _, config = create_bc_data(tmp_path)
    state = train_dapg.initialize_training(config)
    demos = train_dapg.load_demonstrations(state)
    train_dapg.train_iteration(state, BanditEnv(), demos, steps=8)
    path = tmp_path / "rl.pt"
    save_training_checkpoint(path, state)
    expected = train_dapg.train_iteration(state, BanditEnv(), demos, steps=8)
    restored = load_training_checkpoint(path)
    assert restored.critic.optimizer.state
    restore_rng(restored.rng)
    actual = train_dapg.train_iteration(restored, BanditEnv(), demos, steps=8)

    assert actual == pytest.approx(expected, abs=0, rel=0)
    assert restored.iteration == 2
    assert restored.env_steps == 16
    for model, other in ((state.actor, restored.actor), (state.critic, restored.critic)):
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, other.state_dict()[name], rtol=0, atol=0)


def test_train_run_records_evaluation_and_resumes_to_exact_budget(tmp_path, monkeypatch):
    _, config = create_bc_data(tmp_path)
    environments = []

    def create_env(*args, **kwargs):
        env = BanditEnv()
        environments.append(env)
        return env, {}

    monkeypatch.setattr(train_dapg, "create_evaluation_env", create_env)
    run_dir = train_dapg.run(replace(config, total_steps=13))
    saved = load_training_checkpoint(run_dir / "checkpoint.pt")
    assert saved.env_steps == 13
    assert saved.iteration == 2
    assert all(env.closed for env in environments)
    rows = [json.loads(row) for row in (run_dir / "metrics.jsonl").read_text().splitlines()]
    metrics = [row for row in rows if "train/env_steps" in row]
    assert [row["train/env_steps"] for row in metrics] == [8, 13]
    assert all("eval/mean_return" in row for row in metrics)
    resumed_dir = train_dapg.run(
        DAPGConfig(
            resume=run_dir / "checkpoint.pt",
            total_steps=17,
            output_dir=config.output_dir,
        )
    )
    resumed = load_training_checkpoint(resumed_dir / "checkpoint.pt")
    assert resumed.env_steps == 17
    assert resumed.iteration == 3
    assert resumed.config.data_dir == config.data_dir
    assert resumed.config.npg == config.npg
    assert resumed_dir != run_dir


def test_training_improves_known_bandit_objective():
    torch.manual_seed(10)
    policy = GaussianPolicy(1, 1, 1, hidden_dims=())
    with torch.no_grad():
        policy.mean_head.weight.zero_()
        policy.mean_head.bias.zero_()
        policy.log_std.fill_(-0.5)
    normalizer = Normalizer(*(np.array([x], dtype=np.float32) for x in (0, 1, 0, 1)))
    config = DAPGConfig(critic_epochs=2, critic_n_layers=0, critic_batch_size=32, demo_weight=0.1)
    state = TrainingState(
        policy,
        ValueCritic(1, 0, 8, 0.01),
        normalizer,
        {"architecture": {"obs_horizon": 1}},
        config,
    )
    demos = Demonstrations(torch.ones(32, 1), torch.full((32, 1, 1), 0.5))
    before = deepcopy(policy)
    for _ in range(6):
        metrics = train_dapg.train_iteration(state, BanditEnv(), demos, steps=64)
        assert metrics["policy/kl"] <= config.npg.max_kl
    observations = torch.ones(1, 1)
    assert abs(policy(observations).mean.item() - 0.5) < abs(before(observations).mean.item() - 0.5)


def test_training_step_uses_real_mujoco_environment():
    from mujoco_lab.learning.evaluate_cube import create_cube_evaluation_env

    metadata = {
        "architecture": {"physics_steps_per_action": 1, "obs_horizon": 1},
        "dataset_metadata": {
            "replay": {
                "robot": "panda",
                "robot_name": "panda",
                "environment": "table_shelf",
                "cubes": 1,
                "dt": 0.002,
                "physics_steps_per_action": 1,
                "gripper_action_units": "opening_width_m",
                "cube_yaw_range_degrees": 0.0,
            }
        },
    }
    env, _ = create_cube_evaluation_env(metadata, xy_range=0, min_gap=0.01, max_steps=4)
    try:
        obs, _ = env.reset(seed=0)
        action = (env.action_space.low + env.action_space.high) / 2
        policy = GaussianPolicy(len(obs), len(action), 1, hidden_dims=(8,))
        critic = ValueCritic(len(obs), 1, 8, 1e-3)
        normalizer = Normalizer(np.zeros_like(obs), np.ones_like(obs), action, np.ones_like(action))
        config = DAPGConfig(critic_epochs=1, npg=NPGConfig(cg_iterations=2, backtrack_steps=4))
        state = TrainingState(policy, critic, normalizer, metadata, config)
        demos = Demonstrations(torch.from_numpy(obs).repeat(4, 1), torch.zeros(4, 1, len(action)))
        metrics = train_dapg.train_iteration(state, env, demos, steps=4)
        assert state.env_steps == 4
        assert state.iteration == 1
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        env.close()
