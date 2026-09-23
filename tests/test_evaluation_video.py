"""Streaming evaluation video lifecycle and file-based W&B logging."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation import evaluate_policy
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.policies.mse import MSEPolicy


class Env:
    def __init__(self, *, fail_step=False):
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = SimpleNamespace(
            shape=(1,), low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )
        self.task = SimpleNamespace(
            starts=np.array([[0.0, 0.0, 0.0]]),
            goals=np.array([[0.1, 0.1, 0.0]]),
        )
        self.simulator = SimpleNamespace()
        self.steps = 0
        self.fail_step = fail_step

    def reset(self, *, seed):
        self.steps = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("step failed")
        self.steps += 1
        return (
            np.array([0.0], dtype=np.float32),
            0.0,
            False,
            self.steps == 2,
            {
                "success": False,
                "termination_reason": "time_limit" if self.steps == 2 else None,
            },
        )


def run(env, *, video_dir=None, num_video_episodes=0, on_episode=None):
    model = MSEPolicy(1, 1, 1, hidden_dims=())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    stats = Normalizer(
        state_mean=np.zeros(1),
        state_std=np.ones(1),
        action_mean=np.zeros(1),
        action_std=np.ones(1),
    )
    metadata = {
        "architecture": {
            "frame_dim": 1,
            "state_dim": 1,
            "action_dim": 1,
            "obs_horizon": 1,
            "chunk_size": 1,
            "execution_horizon": 1,
            "physics_steps_per_action": 1,
        },
        "train_config": {},
    }
    return evaluate_policy(
        env,
        model,
        stats,
        metadata,
        num_episodes=3,
        seed=100,
        policy_seed=200,
        max_steps=2,
        dt=0.002,
        device=torch.device("cpu"),
        flow_num_steps=1,
        video_dir=video_dir,
        num_video_episodes=num_video_episodes,
        video_fps=20,
        video_width=64,
        video_height=48,
        on_episode=on_episode,
    )


def test_records_first_seeds_and_closes_before_callback(tmp_path, monkeypatch):
    events = []
    camera_lookats = []

    class MovingEnv(Env):
        def reset(self, *, seed):
            self.task.starts[0, 0] = seed / 1000
            return super().reset(seed=seed)

    class Recorder:
        def __init__(self, simulator, output, **settings):
            self.output = Path(output)
            events.append(("create", self.output.name, settings))

        def __enter__(self):
            events.append(("open", self.output.name))
            return self

        def __exit__(self, *args):
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_bytes(b"mp4")
            events.append(("close", self.output.name))

        def record_initial(self, camera):
            camera_lookats.append(camera.lookat.copy())
            events.append(("initial", self.output.name))

        def record_due(self, camera):
            events.append(("due", self.output.name))

    monkeypatch.setattr("mujoco_lab.learning.evaluation.evaluator.VideoRecorder", Recorder)

    def callback(row):
        if "video_path" in row:
            assert Path(row["video_path"]).is_file()
        events.append(("callback", row["env_seed"]))

    rows, summary = run(MovingEnv(), video_dir=tmp_path, num_video_episodes=2, on_episode=callback)
    np.testing.assert_allclose(camera_lookats, [[0.1, 0.05, 0.12], [0.1005, 0.05, 0.12]])
    assert [Path(row["video_path"]).name for row in rows[:2]] == [
        "seed_100.mp4",
        "seed_101.mp4",
    ]
    assert "video_path" not in rows[2]
    assert summary["attempted"] == 3
    assert [event[0] for event in events] == [
        "create",
        "open",
        "initial",
        "due",
        "due",
        "close",
        "callback",
        "create",
        "open",
        "initial",
        "due",
        "due",
        "close",
        "callback",
        "callback",
    ]


def test_video_disabled_never_creates_camera_or_recorder(monkeypatch):
    monkeypatch.setattr(
        "mujoco_lab.learning.evaluation.evaluator.VideoRecorder",
        lambda *a, **k: pytest.fail("recorder created"),
    )
    monkeypatch.setattr(
        "mujoco_lab.learning.evaluation.evaluator.create_free_camera",
        lambda *a, **k: pytest.fail("camera created"),
    )
    rows, _ = run(Env())
    assert all("video_path" not in row for row in rows)


def test_video_closes_when_environment_step_fails(tmp_path, monkeypatch):
    events = []

    class Recorder:
        def __init__(self, simulator, output, **settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("close")

        def record_initial(self, camera):
            events.append("initial")

    monkeypatch.setattr("mujoco_lab.learning.evaluation.evaluator.VideoRecorder", Recorder)
    with pytest.raises(RuntimeError, match="step failed"):
        run(Env(fail_step=True), video_dir=tmp_path, num_video_episodes=1)
    assert events == ["initial", "close"]


def test_video_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "seed_100.mp4"
    path.write_bytes(b"existing")
    monkeypatch.setattr(
        "mujoco_lab.learning.evaluation.evaluator.VideoRecorder",
        lambda *a, **k: pytest.fail("recorder created"),
    )
    with pytest.raises(FileExistsError, match="seed_100.mp4"):
        run(Env(), video_dir=tmp_path, num_video_episodes=1)
    assert path.read_bytes() == b"existing"


def test_logger_sends_completed_mp4_path_at_same_optimizer_step(tmp_path, monkeypatch):
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"mp4")
    events = []

    class FakeRun:
        def define_metric(self, *args, **kwargs):
            pass

        def log(self, row, *, step, commit):
            events.append(("log", row, step, commit))

        def finish(self, *, exit_code):
            events.append(("finish", exit_code))

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: FakeRun(),
            Video=lambda path, *, format: (path, format),
        ),
    )
    with Logger(tmp_path / "run", {}, wandb_mode="online") as logger:
        logger.log_video_file("eval/video/0", video_path, step=9)
    assert events[0] == (
        "log",
        {"optimizer_step": 9, "eval/video/0": (str(video_path), "mp4")},
        9,
        False,
    )
    assert events[1] == ("log", {}, 9, True)
    assert events[2] == ("finish", 0)
    assert json.loads((tmp_path / "run" / "metrics.jsonl").read_text()) == {
        "optimizer_step": 9,
        "event": "video",
        "name": "eval/video/0",
        "path": str(video_path),
    }
