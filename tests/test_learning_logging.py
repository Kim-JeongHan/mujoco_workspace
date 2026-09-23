"""Local learning logs, optional W&B media, and inference checkpoints."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.trainers.train_bc import run_training


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


def test_wandb_lifecycle_video_and_artifact_use_optimizer_axis(tmp_path, monkeypatch):
    events = []

    class FakeRun:
        id = "run123"

        def define_metric(self, *args, **kwargs):
            events.append(("define", args, kwargs))

        def log(self, values, *, step, commit):
            events.append(("log", values, step, commit))

        def log_artifact(self, artifact):
            events.append(("artifact", artifact))

        def finish(self, *, exit_code):
            events.append(("finish", exit_code))

    def fake_video(frames, *, fps, format):
        events.append(("video", frames.shape, frames.dtype, fps, format))
        return "video-object"

    class FakeArtifact:
        def __init__(self, *, name, type, metadata):
            self.name, self.type, self.metadata = name, type, metadata

        def add_file(self, path):
            self.path = path

    fake = SimpleNamespace(
        init=lambda **kwargs: (events.append(("init", kwargs)), FakeRun())[1],
        Video=fake_video,
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    with Logger(tmp_path / "run", {}, wandb_mode="online", wandb_project="demo") as logger:
        logger.log({"train/loss": 1.0}, step=7)
        logger.log_video("evaluation/video", frames, step=7, fps=12)
        logger.log_checkpoint(checkpoint, step=7)
        with pytest.raises(ValueError, match="uint8 RGB"):
            logger.log_video("bad", frames.astype(np.float32), step=7)

    assert ("define", ("*",), {"step_metric": "optimizer_step"}) in events
    assert ("video", (2, 3, 8, 10), np.dtype("uint8"), 12, "mp4") in events
    artifact = next(event[1] for event in events if event[0] == "artifact")
    assert artifact.name == "run123-model" and artifact.metadata == {"optimizer_step": 7}
    assert artifact.path == str(checkpoint)
    assert ("finish", 0) in events
    assert [event[1]["optimizer_step"] for event in events if event[0] == "log" and event[1]] == [
        7,
        7,
    ]
    assert ("log", {}, 7, True) in events


def test_wandb_history_merges_same_step_and_flushes_last_row(tmp_path, monkeypatch, capsys):
    class FakeRun:
        def __init__(self):
            self.pending_step = None
            self.pending = {}
            self.rows = []
            self.calls = []
            self.exit_code = None

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values, *, step, commit):
            self.calls.append((dict(values), step, commit))
            if self.pending_step is not None and step > self.pending_step:
                self.rows.append((self.pending_step, self.pending.copy()))
                self.pending.clear()
            assert self.pending_step is None or step >= self.pending_step
            self.pending_step = step
            self.pending.update(values)
            if commit:
                self.rows.append((step, self.pending.copy()))
                self.pending.clear()
                self.pending_step = None

        def finish(self, *, exit_code):
            self.exit_code = exit_code

    run = FakeRun()
    videos = []

    def fake_video(path, *, format, caption):
        videos.append((path, format, caption))
        return (path, caption)

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: run, Video=fake_video),
    )
    first = tmp_path / "first.mp4"
    third = tmp_path / "third.mp4"
    first.write_bytes(b"mp4")
    third.write_bytes(b"mp4")
    episodes = [
        {"env_seed": 100, "video_path": str(first)},
        {"env_seed": 101},
        {"env_seed": 102, "video_path": str(third)},
    ]
    with Logger(tmp_path / "run", {}, wandb_mode="online") as logger:
        logger.log({"train/loss": 1.0}, step=4)
        logger.log_evaluation({"eval/success_rate": 0.0}, episodes, step=4)
        logger.log({"validation/loss": 0.5}, step=4)
        logger.log({"train/loss": 0.2}, step=8)
        with pytest.raises(ValueError, match="nondecreasing"):
            logger.log({"train/loss": 9.0}, step=7)
    assert [step for step, _ in run.rows] == [4, 8]
    assert set(run.rows[0][1]) == {
        "optimizer_step",
        "train/loss",
        "eval/success_rate",
        "eval/rollout_ep0",
        "eval/rollout_ep2",
        "validation/loss",
    }
    assert run.rows[1][1]["train/loss"] == 0.2
    assert run.calls[-1] == ({}, 8, True)
    assert all(call[2] is False for call in run.calls[:-1])
    assert "env seed 100, optimizer step 4" in videos[0][2]
    assert "env seed 102, optimizer step 4" in videos[1][2]
    assert run.exit_code == 0
    rows = [
        json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6
    assert [row["name"] for row in rows if row.get("event") == "video"] == [
        "eval/rollout_ep0",
        "eval/rollout_ep2",
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[INFO]" in captured.err and '"eval/success_rate": 0.0' in captured.err


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


def test_wandb_flush_error_marks_run_failed_and_survives_finish_error(tmp_path, monkeypatch):
    finished = []

    def fake_log(values, *, step, commit):
        if commit:
            raise RuntimeError("flush failed")

    def fake_finish(*, exit_code):
        finished.append(exit_code)
        raise RuntimeError("finish failed")

    run = SimpleNamespace(
        define_metric=lambda *args, **kwargs: None,
        log=fake_log,
        finish=fake_finish,
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run))
    with (
        pytest.raises(RuntimeError, match="flush failed"),
        Logger(tmp_path / "flush-error", {}, wandb_mode="online") as logger,
    ):
        logger.log({"train/loss": 1.0}, step=3)
    assert finished == [1]


def test_wandb_defaults_online_without_starting_a_real_run(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    init_args = []
    fake_run = SimpleNamespace(
        id="run123",
        define_metric=lambda *args, **kwargs: None,
        finish=lambda **kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: (init_args.append(kwargs), fake_run)[1]),
    )

    with Logger(tmp_path / "default-online", {}) as logger:
        assert logger.wandb_mode == "online"
    assert init_args[0]["mode"] == "online"
    assert init_args[0]["force"] is True


def test_logger_uses_wandb_environment_when_arguments_are_omitted(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_PROJECT", "research")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_RUN_GROUP", "trial")
    with Logger(tmp_path / "env", {}) as logger:
        assert logger.wandb_mode == "disabled"
        assert logger.wandb_project == "research"
        assert logger.wandb_entity == "team"
        assert logger.wandb_group == "trial"


def test_offline_wandb_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="wandb_mode must be online or disabled"):
        Logger(tmp_path / "offline", {}, wandb_mode="offline")  # ty: ignore[invalid-argument-type]


def test_logger_marks_exception_failed_without_swallowing_it(tmp_path, monkeypatch):
    finished = []
    fake_run = SimpleNamespace(
        id="run123",
        define_metric=lambda *args, **kwargs: None,
        finish=lambda *, exit_code: finished.append(exit_code),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: fake_run))
    with (
        pytest.raises(RuntimeError, match="training failed"),
        Logger(tmp_path / "run", {}, wandb_mode="online"),
    ):
        raise RuntimeError("training failed")
    assert finished == [1]


def test_wandb_setup_failure_finishes_started_run(tmp_path, monkeypatch):
    finished = []

    def fail_define(*args, **kwargs):
        raise RuntimeError("metric setup failed")

    fake_run = SimpleNamespace(
        define_metric=fail_define,
        finish=lambda *, exit_code: finished.append(exit_code),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: fake_run))
    with (
        pytest.raises(RuntimeError, match="metric setup failed"),
        Logger(tmp_path / "run", {}, wandb_mode="online"),
    ):
        pass
    assert finished == [1]


def test_checkpoint_round_trip_is_weights_only_and_preserves_normalizer(tmp_path):
    config = TrainConfig(
        robot="forte",
        policy_type="mse",
        hidden_dims=(8,),
        obs_horizon=2,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
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


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_training_logs_true_held_out_loss_with_training_only_stats(
    tmp_path, monkeypatch, policy_type
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train = Episode(states=np.zeros((3, 54), dtype=np.float32), actions=np.zeros((2, 8)))
    validation = Episode(
        states=np.full((3, 54), 1000.0, dtype=np.float32),
        actions=np.ones((2, 8), dtype=np.float32),
    )
    config = TrainConfig(
        policy_type=policy_type,
        hidden_dims=(8,),
        num_epochs=1,
        batch_size=2,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        log_interval=1,
    )
    with Logger(tmp_path / "run", {}, wandb_mode="disabled") as logger:
        model, normalizer = run_training(config, [train], [validation], logger=logger)
    metric_lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in metric_lines]
    assert rows[0]["optimizer_step"] == rows[1]["optimizer_step"] == 1
    assert "train/loss" in rows[0] and "validation/loss" in rows[1]
    assert rows[0]["epoch"] == rows[1]["epoch"] == 1
    assert rows[0]["lr"] == config.lr and rows[0]["elapsed_seconds"] >= 0
    assert np.max(normalizer.state_mean) == 0
    assert model.state_dim == 108
