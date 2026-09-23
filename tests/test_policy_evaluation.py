"""Closed-loop evaluation with small policies and deterministic stand-in environments."""

import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import EvalConfig, TrainConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluate import _scene_settings, _verify_replay, run
from mujoco_lab.learning.evaluation import evaluate_policy, validate_contract
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.simulation import Simulator


class TinyEnv:
    def __init__(self, *, low=-10.0, high=10.0, finish=3, success=False):
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = SimpleNamespace(shape=(1,), low=np.array([low]), high=np.array([high]))
        self.finish = finish
        self.wins = success
        self.actions = []
        self.seeds = []
        self.steps = 0

    def reset(self, *, seed):
        self.steps = 0
        self.seeds.append(seed)
        return np.array([2.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(float(action[0]))
        self.steps += 1
        ended = self.steps == self.finish
        success = ended and self.wins
        return (
            np.array([2.0 + self.steps], dtype=np.float32),
            float(success),
            success,
            ended and not success,
            {"success": success, "termination_reason": "success" if success else "time_limit"},
        )


def checkpoint(tmp_path, *, policy_type="mse", chunk_size=2, execution_horizon=2):
    config = TrainConfig(
        robot="forte",
        policy_type=policy_type,
        hidden_dims=(),
        obs_horizon=2,
        chunk_size=chunk_size,
        execution_horizon=execution_horizon,
        physics_steps_per_action=1,
    )
    model = build_policy(
        policy_type, state_dim=2, action_dim=1, chunk_size=chunk_size, hidden_dims=()
    )
    normalizer = Normalizer(
        state_mean=np.array([1.0], dtype=np.float32),
        state_std=np.array([2.0], dtype=np.float32),
        action_mean=np.array([10.0], dtype=np.float32),
        action_std=np.array([2.0], dtype=np.float32),
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, normalizer, config, optimizer_step=7)
    return load_checkpoint(path)


def evaluate(
    env, model, normalizer, metadata, *, num_episodes=1, seed=50, policy_seed=70, max_steps=5
):
    return evaluate_policy(
        env,
        model,
        normalizer,
        metadata,
        num_episodes=num_episodes,
        seed=seed,
        policy_seed=policy_seed,
        max_steps=max_steps,
        dt=0.002,
        device=torch.device("cpu"),
        flow_num_steps=3,
    )


def test_mse_history_padding_chunk_alignment_clipping_and_success(tmp_path):
    model, stats, metadata = checkpoint(tmp_path)
    policy_inputs = []
    handle = model.net[0].register_forward_pre_hook(
        lambda module, args: policy_inputs.append(args[0].detach().cpu().numpy().copy())
    )
    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        model.net[0].bias.zero_()
    env = TinyEnv(low=0, high=11, finish=3, success=True)
    rows, summary = evaluate(env, model, stats, metadata, num_episodes=2)
    handle.remove()
    # First padded history [2,2] normalizes to [0.5,0.5], hence 10+2*0.5=11.
    # At the next chunk boundary history [3,4] gives the current-frame action 13,
    # clipped to 11; a fresh chunk is not sampled after the success step.
    assert env.actions == [11, 11, 11] * 2
    np.testing.assert_allclose(np.concatenate(policy_inputs), [[0.5, 0.5], [1.0, 1.5]] * 2)
    assert [row["steps"] for row in rows] == [3, 3]
    assert all(row["success"] for row in rows)
    assert all(row["termination_reason"] == "success" for row in rows)
    assert [row["clipped_actions"] for row in rows] == [1, 1]
    assert [row["clipped_action_axes"] for row in rows] == [[1], [1]]
    assert summary["clipped_action_axes"] == [2]
    assert summary["action_clip_axis_fraction"] == pytest.approx([1 / 3])
    assert summary["action_overrun_mean"] == pytest.approx([2 / 3])
    assert summary["action_overrun_max"] == pytest.approx([2])
    assert rows[0]["sim_seconds"] == pytest.approx(0.006)
    assert summary["action_clip_fraction"] == pytest.approx(1 / 3)
    assert summary["mean_success_sim_seconds"] == pytest.approx(0.006)


def test_nonfinite_action_fails_attempt_without_stepping_and_timeout_counts(tmp_path):
    model, stats, metadata = checkpoint(tmp_path)
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].bias.fill_(float("nan"))
    env = TinyEnv()
    rows, summary = evaluate(env, model, stats, metadata, num_episodes=2)
    assert env.seeds == [50, 51]
    assert env.actions == []
    assert [row["policy_seed"] for row in rows] == [70, 71]
    assert summary["attempted"] == summary["nonfinite_failures"] == 2
    assert summary["mean_success_sim_seconds"] is None
    with torch.no_grad():
        model.net[0].bias.zero_()
    rows, summary = evaluate(TinyEnv(finish=2), model, stats, metadata)
    assert rows[0]["termination_reason"] == "time_limit"
    assert not rows[0]["success"] and summary["timeouts"] == 1


@pytest.mark.parametrize("max_steps", [1, 5])
@pytest.mark.parametrize("failure_reason", [None, "failed_stack"])
def test_terminated_failure_is_not_a_timeout(tmp_path, max_steps, failure_reason):
    model, stats, metadata = checkpoint(tmp_path)

    class FailedEnv(TinyEnv):
        def step(self, action):
            obs, reward, _, _, _ = super().step(action)
            return (
                obs,
                reward,
                True,
                False,
                {
                    "success": np.bool_(False),
                    "termination_reason": failure_reason,
                },
            )

    rows, summary = evaluate(FailedEnv(), model, stats, metadata, max_steps=max_steps)
    assert rows[0]["termination_reason"] == (failure_reason or "terminated")
    assert not rows[0]["success"]
    assert summary["timeouts"] == 0


def test_flow_seed_reproducibility_and_rng_and_mode_restoration(tmp_path):
    model, stats, metadata = checkpoint(tmp_path, policy_type="flow")
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].bias.zero_()
    model.train()
    rng_before = torch.random.get_rng_state().clone()
    env1, env2 = TinyEnv(), TinyEnv()
    rows1, _ = evaluate(env1, model, stats, metadata, num_episodes=2)
    rows2, _ = evaluate(env2, model, stats, metadata, num_episodes=2)
    assert rows1 == rows2
    assert env1.actions == env2.actions
    assert env1.seeds == env2.seeds == [50, 51]
    assert model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    different = TinyEnv()
    evaluate(different, model, stats, metadata, policy_seed=71)
    assert different.actions != env1.actions[: len(different.actions)]


def test_contract_rejects_bad_timing_dimensions_and_stats(tmp_path):
    model, stats, metadata = checkpoint(tmp_path)
    env = TinyEnv()
    metadata["architecture"]["physics_steps_per_action"] = 2
    metadata["train_config"]["physics_steps_per_action"] = 2
    with pytest.raises(ValueError, match="dataset replay physics_steps_per_action"):
        validate_contract(model, stats, metadata, env)
    metadata["architecture"]["physics_steps_per_action"] = 1
    metadata["train_config"]["physics_steps_per_action"] = 1
    env.observation_space.shape = (2,)
    with pytest.raises(ValueError, match="observation dimension"):
        validate_contract(model, stats, metadata, env)
    env.observation_space.shape = (1,)
    stats.state_std[0] = 0
    with pytest.raises(ValueError, match="state_std must be positive"):
        validate_contract(model, stats, metadata, env)


def test_scene_settings_prefer_replay_and_reject_conflicts():
    metadata = {
        "train_config": {"robot": "panda", "cubes": 2, "environment": "table_shelf"},
        "dataset_metadata": {
            "replay": {"robot": "forte", "cubes": 2, "environment": "table_shelf", "dt": 0.002}
        },
    }
    with pytest.raises(ValueError, match="replay robot differs"):
        _scene_settings(metadata)
    metadata["train_config"]["robot"] = "forte"
    robot, robot_name, cubes, environment, dt, sources = _scene_settings(metadata)
    assert (robot, robot_name, cubes, environment, dt) == (
        "forte",
        "forte",
        2,
        "table_shelf",
        0.002,
    )
    assert sources["robot"] == "dataset_metadata.replay"
    assert sources["robot_name"] == "robot"


def test_replay_rejects_physics_change_with_same_visual_hash(monkeypatch):
    simulator = SimpleNamespace(dt=0.004)
    metadata = {
        "dataset_metadata": {
            "replay": {
                "schema_version": 2,
                "model_sha256": "old-physics",
                "visual_sha256": "same-visual",
                "mujoco_version": "3.13.0",
            }
        }
    }
    monkeypatch.setattr(
        "mujoco_lab.learning.evaluate.cube_stack_metadata",
        lambda *args, **kwargs: {
            "model_sha256": "new-physics",
            "visual_sha256": "same-visual",
            "mujoco_version": "3.13.0",
        },
    )
    with pytest.raises(ValueError, match="model_sha256"):
        _verify_replay(cast(Simulator, simulator), metadata, robot="forte", cubes=2, dt=0.004)


def test_cli_run_writes_fresh_episode_summary_and_aggregate_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    model, stats, metadata = checkpoint(tmp_path)
    metadata["dataset_metadata"] = {
        "replay": {
            "robot": "forte",
            "robot_name": "arm",
            "cubes": 2,
            "environment": "table_shelf",
            "dt": 0.004,
        },
        "train_seeds": [50],
    }

    class Robot:
        def change_controller(self, controller):
            self.controller = controller

    class FakeSimulator:
        def __init__(self, scene, *, robots, dt):
            assert robots[0].name == "arm" and robots[0].robot_type == "forte"
            assert dt == 0.004
            self.robots = {"arm": Robot()}

    monkeypatch.setattr(
        "mujoco_lab.learning.evaluate.load_checkpoint", lambda path: (model, stats, metadata)
    )
    monkeypatch.setattr("mujoco_lab.learning.evaluate.create_cube_stack", lambda *a, **k: None)
    monkeypatch.setattr("mujoco_lab.learning.evaluate.Simulator", FakeSimulator)
    monkeypatch.setattr("mujoco_lab.learning.evaluate._verify_replay", lambda *a, **k: None)
    monkeypatch.setattr("mujoco_lab.learning.evaluate.create_controller", lambda *a, **k: None)
    monkeypatch.setattr("mujoco_lab.learning.evaluate.CubeStackTask", lambda *a, **k: None)
    monkeypatch.setattr(
        "mujoco_lab.learning.evaluate.CubeStackEnv",
        lambda *a, **k: TinyEnv(finish=1, success=True),
    )
    run_dir, summary = run(
        EvalConfig(
            checkpoint=tmp_path / "checkpoint.pt",
            device="cpu",
            num_episodes=2,
            num_video_episodes=0,
            seed=50,
            max_steps=1,
            output_dir=tmp_path / "logs",
        )
    )
    assert run_dir.parent == tmp_path / "logs" / "eval" / "mse"
    assert summary["attempted"] == summary["successes"] == 2
    assert json.loads((run_dir / "config.json").read_text())["effective"]["seed_overlaps"] == {
        "train": [50],
        "validation": [],
        "test": [],
    }
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
    assert [row["env_seed"] for row in rows] == [50, 51]
    assert json.loads((run_dir / "summary.json").read_text()) == summary
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics == [
        {
            "optimizer_step": 7,
            "eval/attempted": 2,
            "eval/successes": 2,
            "eval/success_rate": 1.0,
            "eval/timeouts": 0,
            "eval/nonfinite_failures": 0,
            "eval/action_clip_fraction": summary["action_clip_fraction"],
            "eval/mean_success_sim_seconds": 0.004,
        }
    ]
