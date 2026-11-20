"""BC initialization, demonstration loading, full training, and exact boundary resume."""

import json
import math
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces
from metadata_config import checkpoint_environment

from mujoco_lab.learning.agents import DAPGAgent, Demonstrations
from mujoco_lab.learning.checkpoint import (
    DAPGTrainingState,
    load_training_checkpoint,
    restore_rng,
    save_checkpoint,
    save_training_checkpoint,
)
from mujoco_lab.learning.config import CriticConfig, MSEBCTrainConfig
from mujoco_lab.learning.config.dapg import DAPGConfig
from mujoco_lab.learning.config.npg import NPGConfig
from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.metadata import (
    CheckpointMetadata,
    DatasetMetadata,
    EnvironmentMetadata,
    ObservationMetadata,
    PolicyMetadata,
)
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.trainers import train_dapg


class BanditEnv:
    """A one-step quadratic task with a known optimal action of 0.5."""

    def __init__(self):
        self.action_space = spaces.Box(-2, 2, (1,), dtype=np.float32)
        self.max_steps = 2
        self.action_dt = 0.002
        self.simulator = SimpleNamespace(data=SimpleNamespace(time=0.0))
        self.task = None
        self.closed = False

    def reset(self, *, seed=None):
        self.steps = 0
        self.simulator.data.time = 0.0
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        self.simulator.data.time += self.action_dt
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
    config = MSEBCTrainConfig(
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
        "cube_yaw_range_degrees": 0.0,
        "xy_range": 0.02,
        "min_gap": 0.01,
    }
    observation = {"frame_dim": 1, "rotation_indices": []}
    path = tmp_path / "bc.pt"
    save_checkpoint(
        path,
        model,
        normalizer,
        config,
        optimizer_step=12,
        dataset=DatasetMetadata(train_seeds=(0,)),
        environment=EnvironmentMetadata.from_recording(replay),
        observation=ObservationMetadata.from_dict(observation),
    )
    data_dir = tmp_path / "demos"
    data_dir.mkdir()
    for seed in range(2):
        Episode(
            states=np.ones((5, 1), dtype=np.float32),
            actions=np.full((4, 1), 0.5, dtype=np.float32),
            metadata={
                "success": True,
                "seed": seed,
                "replay": replay,
                "observation": observation,
            },
        ).save(data_dir / f"episode_{seed:06d}.npz")
    return model, DAPGConfig(
        init_from=path,
        data_dir=data_dir,
        output_dir=tmp_path / "log",
        device="cpu",
        total_steps=16,
        rollout_trajectories=8,
        critic=CriticConfig(hidden_dims=(8, 4), num_epochs=1, batch_size=4),
        eval_interval_iterations=1,
        eval_episodes=1,
        eval_video_episodes=0,
        checkpoint_interval_iterations=1,
        npg=NPGConfig(cg_iterations=4),
    )


def test_initialization_keeps_first_bc_action_and_observation_history(tmp_path):
    bc, config = create_bc_data(tmp_path)
    state = train_dapg.initialize_training(config)
    observations = torch.randn(5, 2)
    # Slicing the output head changes GEMM width, allowing float32 roundoff.
    torch.testing.assert_close(state.agent.policy(observations).mean, bc(observations)[:, :1])
    for name, tensor in state.agent.policy.mean_head.state_dict().items():
        torch.testing.assert_close(tensor, bc.net[-1].state_dict()[name][:1], rtol=0, atol=0)
    assert state.agent.policy.chunk_size == 1
    assert state.metadata.policy.execution_horizon == 1
    assert state.metadata.source_bc is not None
    assert state.metadata.source_bc.policy.chunk_size == 3
    assert state.metadata.policy.obs_horizon == 2
    demos = train_dapg.load_demonstrations(state)
    # The held-out seed 1 must not become demonstration supervision.
    assert demos.states.shape == (4, 2)
    assert demos.actions.shape == (4, 1, 1)


@pytest.mark.parametrize("scale_factor", [1.0, 3.0])
def test_initial_exploration_is_independent_of_dataset_action_scale(tmp_path, scale_factor):
    bc = MSEPolicy(2, 8, 3, hidden_dims=(8,))
    normalizer = Normalizer(
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.03], dtype=np.float32) * scale_factor,
    )
    path = tmp_path / "bc.pt"
    save_checkpoint(
        path,
        bc,
        normalizer,
        MSEBCTrainConfig(hidden_dims=(8,), chunk_size=3),
        optimizer_step=0,
        environment=checkpoint_environment(MSEBCTrainConfig(hidden_dims=(8,), chunk_size=3)),
    )
    state = train_dapg.initialize_training(DAPGConfig(init_from=path, device="cpu"))
    physical_std = state.agent.policy.log_std.detach().exp()[0] * torch.from_numpy(
        normalizer.action_std
    )
    torch.testing.assert_close(physical_std, torch.tensor([math.radians(0.5)] * 7 + [0.0005]))
    observations = torch.randn(5, 2)
    torch.testing.assert_close(state.agent.policy(observations).mean, bc(observations)[:, :1])


def test_checkpoint_resume_reproduces_next_training_iteration(tmp_path):
    _, config = create_bc_data(tmp_path)
    state = train_dapg.initialize_training(config)
    demos = train_dapg.load_demonstrations(state)
    train_dapg.train_iteration(state, BanditEnv(), demos)
    path = tmp_path / "rl.pt"
    save_training_checkpoint(path, state)
    expected = train_dapg.train_iteration(state, BanditEnv(), demos)
    restored = load_training_checkpoint(path)
    assert restored.agent.config.critic.hidden_dims == (8, 4)
    assert restored.agent.critic.optimizer.state
    assert restored.rng is not None
    restore_rng(restored.rng)
    actual = train_dapg.train_iteration(restored, BanditEnv(), demos)

    assert actual == pytest.approx(expected, abs=0, rel=0)
    assert restored.agent.iteration == 2
    assert restored.env_steps == 16
    for model, other in (
        (state.agent.policy, restored.agent.policy),
        (state.agent.critic, restored.agent.critic),
    ):
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, other.state_dict()[name], rtol=0, atol=0)


def test_state_dict_can_be_restored_repeatedly_without_changing_saved_config(tmp_path):
    _, config = create_bc_data(tmp_path)
    state = train_dapg.initialize_training(config)
    payload = state.state_dict()
    saved_config = deepcopy(payload["config"])

    for _ in range(2):
        restored = DAPGTrainingState.from_state_dict(payload)
        assert restored.agent.config == config
        for name, tensor in state.agent.policy.state_dict().items():
            torch.testing.assert_close(tensor, restored.agent.policy.state_dict()[name])
        assert payload["config"] == saved_config


def test_training_bootstraps_critic_at_episode_time_limits(tmp_path, monkeypatch):
    _, config = create_bc_data(tmp_path)
    config = replace(config, rollout_trajectories=1)
    state = train_dapg.initialize_training(config)
    demos = train_dapg.load_demonstrations(state)
    targets = []

    class TimeoutEnv(BanditEnv):
        def step(self, action):
            self.steps += 1
            return (
                np.array([1.0], dtype=np.float32),
                0.0,
                False,
                self.steps == self.max_steps,
                {"success": False},
            )

    monkeypatch.setattr(
        state.agent.critic, "forward", lambda observations: torch.ones(len(observations))
    )

    def record_update(observations, returns):
        targets.extend(returns.tolist())
        return {"Baseline Loss": 0.0}

    monkeypatch.setattr(state.agent.critic, "update", record_update)
    train_dapg.train_iteration(state, TimeoutEnv(), demos)
    assert sorted(targets) == pytest.approx(sorted([config.gamma**2, config.gamma]))


def test_train_run_finishes_trajectory_batches_and_resumes(tmp_path, monkeypatch):
    _, config = create_bc_data(tmp_path)
    config.eval_interval_iterations = 3
    environments = []

    def create_env(*args, **kwargs):
        env = BanditEnv()
        environments.append(env)
        return env, {}

    monkeypatch.setattr(train_dapg, "create_evaluation_env", create_env)
    run_dir = train_dapg.run(replace(config, total_steps=13))
    saved = load_training_checkpoint(run_dir / "checkpoint.pt")
    assert saved.env_steps == 16
    assert saved.agent.iteration == 2
    assert all(env.closed for env in environments)
    rows = [json.loads(row) for row in (run_dir / "metrics.jsonl").read_text().splitlines()]
    metrics = [row for row in rows if "rollout/steps" in row]
    assert [row["train/env_steps"] for row in metrics] == [8, 16]
    assert all(row["rollout/completed_episodes"] == 8 for row in metrics)
    evaluations = [row for row in metrics if "eval/mean_return" in row]
    assert evaluations[-1]["train/env_steps"] == 16
    assert [row["train/iteration"] for row in metrics] == [1, 2]
    assert all(row["elapsed_seconds"] >= 0 for row in metrics)
    reports = sorted((run_dir / "eval").glob("*/summary.json"))
    assert len(reports) == 1
    for report, row in zip(reports, evaluations, strict=True):
        summary = json.loads(report.read_text())
        assert summary["attempted"] == row["eval/attempted"] == 1
        assert summary["success_rate"] == row["eval/success_rate"]
        episodes = [
            json.loads(line) for line in (report.parent / "episodes.jsonl").read_text().splitlines()
        ]
        assert [episode["env_seed"] for episode in episodes] == [config.eval_seed]
    resumed_dir = train_dapg.run(
        DAPGConfig(
            resume=run_dir / "checkpoint.pt",
            total_steps=17,
            output_dir=config.output_dir,
        )
    )
    resumed = load_training_checkpoint(resumed_dir / "checkpoint.pt")
    assert resumed.env_steps == 24
    assert resumed.agent.iteration == 3
    assert resumed.agent.config.data_dir == config.data_dir
    assert resumed.agent.config.npg == config.npg
    assert resumed.agent.config.rollout_trajectories == 8
    assert resumed_dir != run_dir


def test_training_improves_known_bandit_objective():
    torch.manual_seed(10)
    policy = GaussianPolicy(1, 1, 1, hidden_dims=())
    with torch.no_grad():
        for parameter in policy.mean_head.parameters():
            parameter.zero_()
        policy.log_std.fill_(-0.5)
    normalizer = Normalizer(*(np.array([x], dtype=np.float32) for x in (0, 1, 0, 1)))
    config = DAPGConfig(
        device="cpu",
        rollout_trajectories=64,
        critic=CriticConfig(hidden_dims=(), num_epochs=2, batch_size=32),
        demo_weight=0.1,
    )
    state = DAPGTrainingState(
        DAPGAgent(policy, ValueCritic(1, (), 0.01), config),
        normalizer,
        CheckpointMetadata(
            PolicyMetadata("gaussian", 1, 1),
            ObservationMetadata(1),
            environment=checkpoint_environment(MSEBCTrainConfig(action_execution_hz=500)),
        ),
    )
    demos = Demonstrations(torch.ones(32, 1), torch.full((32, 1, 1), 0.5))
    before = deepcopy(policy)
    for _ in range(6):
        metrics = train_dapg.train_iteration(state, BanditEnv(), demos)
        assert metrics["policy/kl"] <= config.npg.max_kl
    observations = torch.ones(1, 1)
    assert abs(policy(observations).mean.item() - 0.5) < abs(before(observations).mean.item() - 0.5)


def test_training_step_uses_real_mujoco_environment():
    from mujoco_lab.learning.evaluate_cube import create_cube_evaluation_env

    environment = EnvironmentMetadata(CubeStackReplayConfig(robot="panda", cubes=1), 0.002, "panda")
    env, _ = create_cube_evaluation_env(environment, xy_range=0, min_gap=0.01, max_steps=4)
    try:
        obs, _ = env.reset(seed=0)
        action = (env.action_space.low + env.action_space.high) / 2
        policy = GaussianPolicy(len(obs), len(action), 1, hidden_dims=(8,))
        critic = ValueCritic(len(obs), (8,), 1e-3)
        normalizer = Normalizer(np.zeros_like(obs), np.ones_like(obs), action, np.ones_like(action))
        config = DAPGConfig(
            device="cpu",
            rollout_trajectories=2,
            critic=CriticConfig(num_epochs=1),
            npg=NPGConfig(cg_iterations=2, backtrack_steps=4),
        )
        metadata = CheckpointMetadata(
            PolicyMetadata("gaussian", len(obs), len(action), hidden_dims=(8,)),
            ObservationMetadata.from_dict(env.observation_metadata),
            environment,
        )
        state = DAPGTrainingState(DAPGAgent(policy, critic, config), normalizer, metadata)
        demos = Demonstrations(torch.from_numpy(obs).repeat(4, 1), torch.zeros(4, 1, len(action)))
        metrics = train_dapg.train_iteration(state, env, demos)
        assert state.env_steps == 8
        assert metrics["rollout/completed_episodes"] == 2
        assert state.agent.iteration == 1
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        env.close()
