"""Local learning logs, optional W&B media, and inference checkpoints."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.policies.factory import build_policy


def test_disabled_logger_writes_dynamic_jsonl_without_mutating_metrics(tmp_path, capsys):
    run_dir = tmp_path / "run"
    row = {"train/loss": 0.5}
    with Logger(run_dir, {"policy": "mse"}, wandb_mode="disabled") as logger:
        logger.log(row, step=1)
        logger.log({"validation/loss": 0.4}, step=1)
        assert row == {"train/loss": 0.5}

    assert json.loads((run_dir / "config.json").read_text()) == {"policy": "mse"}
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines == [
        {"optimizer_step": 1, "train/loss": 0.5},
        {"optimizer_step": 1, "validation/loss": 0.4},
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[INFO]" in captured.err and '"train/loss": 0.5' in captured.err
    with pytest.raises(FileExistsError), Logger(run_dir, {}, wandb_mode="disabled"):
        pass


def test_wandb_history_shares_optimizer_step_and_flushes_on_exit(tmp_path, monkeypatch):
    run = SimpleNamespace(define_metric=Mock(), log=Mock(), finish=Mock())
    video = Mock(return_value="video")
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Video=video)
    )
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"mp4")
    with Logger(tmp_path / "run", {}, wandb_mode="online") as logger:
        logger.log({"train/loss": 1.0}, step=4)
        logger.log_evaluation(
            {"eval/success_rate": 0.5}, [{"env_seed": 100, "video_path": str(path)}], step=4
        )
        logger.log({"validation/loss": 0.5}, step=4)
        logger.log({"train/loss": 0.2}, step=8)
        with pytest.raises(ValueError, match="nondecreasing"):
            logger.log({"train/loss": 9.0}, step=7)
    calls = run.log.call_args_list
    assert [call.kwargs for call in calls] == [
        {"step": 4, "commit": False},
        {"step": 4, "commit": False},
        {"step": 4, "commit": False},
        {"step": 8, "commit": False},
        {"step": 8, "commit": True},
    ]
    assert calls[1].args[0] == {
        "optimizer_step": 4,
        "eval/success_rate": 0.5,
        "eval/rollout_ep0": "video",
    }
    assert calls[-1].args == ({},)
    assert "env seed 100, optimizer step 4" in video.call_args.kwargs["caption"]
    run.finish.assert_called_once_with(exit_code=0)


def test_wandb_flushes_pending_history_before_failed_finish(tmp_path, monkeypatch):
    calls = []
    run = SimpleNamespace(
        define_metric=lambda *args, **kwargs: None,
        log=lambda values, *, step, commit: calls.append((values, step, commit)),
        finish=lambda *, exit_code: calls.append(("finish", exit_code)),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run))
    with (
        pytest.raises(RuntimeError, match="failed"),
        Logger(tmp_path / "failed", {}, wandb_mode="online") as logger,
    ):
        logger.log({"train/loss": 1.0}, step=3)
        raise RuntimeError("failed")
    assert calls == [
        ({"optimizer_step": 3, "train/loss": 1.0}, 3, False),
        ({}, 3, True),
        ("finish", 1),
    ]


def test_checkpoint_round_trip_is_weights_only_and_preserves_normalizer(tmp_path):
    config = TrainConfig(
        robot="forte",
        policy_type="mse",
        hidden_dims=(8,),
        obs_horizon=2,
        chunk_size=1,
        execution_horizon=1,
        action_execution_hz=500,
    )
    model = build_policy("mse", state_dim=108, action_dim=8, chunk_size=1, hidden_dims=(8,))
    normalizer = Normalizer(
        state_mean=np.arange(54, dtype=np.float32),
        state_std=np.ones(54, dtype=np.float32),
        action_mean=np.arange(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, normalizer, config, optimizer_step=12, dataset_metadata={"dt": 0.002}
    )
    raw = torch.load(path, weights_only=True, map_location="cpu")
    assert raw["architecture"]["obs_horizon"] == 2
    loaded, stats, info = load_checkpoint(path)
    np.testing.assert_array_equal(stats.state_mean, normalizer.state_mean)
    assert info["optimizer_step"] == 12 and info["dataset_metadata"]["dt"] == 0.002
    sample = torch.ones((2, 108))
    with torch.no_grad():
        torch.testing.assert_close(model.sample_actions(sample), loaded.sample_actions(sample))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, normalizer, config, optimizer_step=13)
