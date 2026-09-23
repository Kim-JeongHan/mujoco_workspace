"""Periodic training evaluation uses the live policy and preserves optimization."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import checkpoint_metadata
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation import evaluate_policy
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.trainers.train_bc import run_training


class TinyEnv:
    observation_space = SimpleNamespace(shape=(1,))
    action_space = SimpleNamespace(
        shape=(1,), low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
    )

    def reset(self, *, seed):
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([0.0], dtype=np.float32), 0.0, False, True, {"success": False}


def test_evaluation_interval_final_step_and_training_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    demonstration = Episode(
        states=np.arange(6, dtype=np.float32).reshape(-1, 1),
        actions=np.arange(5, dtype=np.float32).reshape(-1, 1),
    )
    config = TrainConfig(
        policy_type="flow",
        hidden_dims=(8,),
        obs_horizon=1,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        batch_size=2,
        num_epochs=2,
        log_interval=100,
        eval_interval=4,
    )
    observed = []

    def evaluate(model, normalizer, step):
        assert model.training
        metadata = checkpoint_metadata(model, normalizer, config, optimizer_step=step)
        rows, summary = evaluate_policy(
            TinyEnv(),
            model,
            normalizer,
            metadata,
            num_episodes=1,
            seed=100,
            policy_seed=200,
            max_steps=1,
            dt=0.002,
            device=torch.device("cpu"),
            flow_num_steps=2,
        )
        assert model.training
        observed.append((step, rows[0]["env_seed"], summary["attempted"]))

    evaluated, _ = run_training(config, [demonstration], evaluate=evaluate)
    reference, _ = run_training(config, [demonstration])
    assert observed == [(4, 100, 1), (6, 100, 1)]
    for name, parameter in evaluated.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name])

    disabled = []
    run_training(
        replace(config, num_epochs=1, eval_interval=0),
        [demonstration],
        evaluate=lambda *_args: disabled.append(True),
    )
    assert disabled == []


def test_training_cli_writes_step_results_video_and_same_step_logs(tmp_path, monkeypatch):
    from mujoco_lab.learning import train as cli

    config = TrainConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "logs",
        robot="forte",
        hidden_dims=(8,),
        obs_horizon=1,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        batch_size=2,
        num_epochs=1,
        eval_interval=1,
        num_eval_episodes=1,
    )
    monkeypatch.setenv("WANDB_MODE", "disabled")
    demonstration = Episode(
        states=np.zeros((3, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        metadata={"seed": 7},
    )
    model = build_policy("mse", state_dim=1, action_dim=1, chunk_size=1, hidden_dims=(8,))
    normalizer = Normalizer(
        state_mean=np.zeros(1),
        state_std=np.ones(1),
        action_mean=np.zeros(1),
        action_std=np.ones(1),
    )
    env = TinyEnv()
    monkeypatch.setattr(cli.tyro, "cli", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(cli, "load_episodes", lambda _path: [demonstration])
    monkeypatch.setattr(
        cli, "split_episodes", lambda _episodes, **_kwargs: ([demonstration], [], [])
    )
    monkeypatch.setattr(cli, "create_evaluation_env", lambda *_args, **_kwargs: (env, 0.002, {}))

    def fake_training(_config, _train, _validation, *, logger, evaluate):
        evaluate(model, normalizer, 1)
        return model, normalizer

    monkeypatch.setattr(cli, "run_training", fake_training)

    def fake_evaluation(_env, current_model, _normalizer, metadata, **kwargs):
        assert current_model is model
        assert metadata["optimizer_step"] == 1
        video = kwargs["video_dir"] / "seed_10000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"mp4")
        kwargs["on_episode"](
            {
                "env_seed": 10000,
                "success": False,
                "steps": 1,
                "termination_reason": "time_limit",
                "video_path": str(video),
            }
        )
        return [{"env_seed": 10000, "video_path": str(video)}], {
            "attempted": 1,
            "successes": 0,
            "success_rate": 0.0,
            "timeouts": 1,
            "nonfinite_failures": 0,
            "action_clip_fraction": 0.0,
            "mean_success_sim_seconds": None,
        }

    monkeypatch.setattr(cli, "evaluate_policy", fake_evaluation)
    cli.main()

    run_dir = next((tmp_path / "logs" / "bc" / "mse").iterdir())
    step_dir = run_dir / "eval" / "step_00000001"
    assert (run_dir / "checkpoint_step_00000001.pt").is_file()
    assert (run_dir / "checkpoint.pt").is_file()
    assert json.loads((step_dir / "summary.json").read_text())["attempted"] == 1
    assert json.loads((step_dir / "episodes.jsonl").read_text())["video_path"].endswith(
        "seed_10000.mp4"
    )
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any(row.get("eval/attempted") == 1 and row["optimizer_step"] == 1 for row in rows)
    assert any(
        row.get("name") == "eval/rollout_ep0"
        and row["env_seed"] == 10000
        and row["optimizer_step"] == 1
        for row in rows
    )
    assert any(row.get("event") == "checkpoint" and row["optimizer_step"] == 1 for row in rows)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"physics_steps_per_action": 0}, "physics_steps_per_action"),
        ({"execution_horizon": 2, "chunk_size": 1}, "execution_horizon"),
    ],
)
def test_training_cli_rejects_invalid_evaluation_timing_before_loading_data(
    tmp_path, monkeypatch, overrides, message
):
    from mujoco_lab.learning import train as cli

    config = replace(TrainConfig(output_dir=tmp_path), **overrides)
    monkeypatch.setattr(cli.tyro, "cli", lambda *_args, **_kwargs: config)

    def unexpected_load(_path):
        pytest.fail("Data must not load before timing validation")

    monkeypatch.setattr(cli, "load_episodes", unexpected_load)
    with pytest.raises(ValueError, match=message):
        cli.main()
