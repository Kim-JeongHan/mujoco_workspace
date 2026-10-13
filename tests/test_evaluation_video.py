"""Streaming evaluation video lifecycle and file-based W&B logging."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.config import EvaluationConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation import PolicyEvaluator
from mujoco_lab.learning.policies.mse import MSEPolicy


class Env:
    physics_steps_per_action = 1

    def __init__(self, *, fail_step=False):
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = SimpleNamespace(
            shape=(1,), low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )
        self.task = SimpleNamespace(
            starts=np.array([[0.0, 0.0, 0.0]]),
            goals=np.array([[0.1, 0.1, 0.0]]),
        )
        self.simulator = SimpleNamespace(dt=0.002, data=SimpleNamespace(time=0.0))
        self.steps = 0
        self.fail_step = fail_step

    @property
    def camera_lookat(self):
        center = (self.task.starts.mean(axis=0) + self.task.goals.mean(axis=0)) / 2
        center[2] += 0.12
        return center

    @property
    def action_dt(self):
        return self.simulator.dt * self.physics_steps_per_action

    def reset(self, *, seed):
        self.steps = 0
        self.simulator.data.time = 0.0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("step failed")
        self.steps += 1
        self.simulator.data.time += self.simulator.dt
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


def run(env, *, video_dir=None, num_video_episodes=0, on_episode=None, viewer=False):
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
    return PolicyEvaluator(
        env,
        EvaluationConfig(
            num_episodes=3,
            seed=100,
            max_seconds=0.004,
            viewer=viewer,
            video_episodes=num_video_episodes,
            video_fps=20,
            video_width=64,
            video_height=48,
        ),
        torch.device("cpu"),
    ).evaluate(
        model, stats, metadata, sampling_num_steps=1, video_dir=video_dir, on_episode=on_episode
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


class Viewer:
    def __init__(self, *, close_after=100):
        self.close_after = close_after
        self.sync_calls = 0
        self.locked = False
        self.closed = False

    def is_running(self):
        return self.sync_calls < self.close_after

    @contextmanager
    def lock(self):
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    def sync(self, *, state_only=False):
        assert state_only
        assert not self.locked
        self.sync_calls += 1


def install_viewer(monkeypatch, viewer):
    @contextmanager
    def viewing(simulator, camera):
        try:
            yield viewer
        finally:
            viewer.closed = True

    monkeypatch.setattr("mujoco_lab.learning.evaluation.evaluator._passive_viewer", viewing)


def test_viewer_preserves_headless_results_and_locks_reset_and_step(monkeypatch):
    viewer = Viewer()
    install_viewer(monkeypatch, viewer)
    waits = []
    clock = [0.0]

    def sleep(duration):
        waits.append(duration)
        clock[0] += duration

    predict = PolicyEvaluator._predict_actions

    def timed_predict(self, *args):
        clock[0] += 0.001
        return predict(self, *args)

    monkeypatch.setattr(PolicyEvaluator, "_predict_actions", timed_predict)
    monkeypatch.setattr("mujoco_lab.learning.evaluation.evaluator.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("mujoco_lab.learning.evaluation.evaluator.time.sleep", sleep)

    class ViewingEnv(Env):
        def __init__(self):
            super().__init__()
            self.simulator.model = SimpleNamespace(opt=SimpleNamespace(timestep=self.simulator.dt))

        def reset(self, *, seed):
            assert viewer.locked
            return super().reset(seed=seed)

        def step(self, action):
            assert viewer.locked
            assert self.simulator.model.opt.timestep == self.simulator.dt
            # Simulate a Physics-panel timestep edit, restored after viewer sync.
            self.simulator.model.opt.timestep = 1.0
            return super().step(action)

    expected = run(Env())
    assert waits == []
    assert run(ViewingEnv(), viewer=True) == expected
    assert viewer.sync_calls == 6
    assert viewer.closed
    np.testing.assert_allclose(waits, [0.001] * 6)


def test_closing_viewer_ends_evaluation_without_starting_another_episode(monkeypatch):
    viewer = Viewer(close_after=1)
    install_viewer(monkeypatch, viewer)
    env = Env()
    env.simulator.model = SimpleNamespace(opt=SimpleNamespace(timestep=env.simulator.dt))
    rows, summary = run(env, viewer=True)
    assert len(rows) == summary["attempted"] == 1
    assert rows[0]["termination_reason"] == "viewer_closed"
    assert rows[0]["steps"] == 1
    assert rows[0]["sim_seconds"] == pytest.approx(env.action_dt)
    assert not rows[0]["success"]
    assert viewer.closed


def test_viewer_and_video_close_when_environment_step_fails(tmp_path, monkeypatch):
    viewer = Viewer()
    install_viewer(monkeypatch, viewer)
    events = []

    class Recorder:
        def __init__(self, simulator, output, **settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("video_closed")

        def record_initial(self, camera):
            pass

    monkeypatch.setattr("mujoco_lab.learning.evaluation.evaluator.VideoRecorder", Recorder)
    with pytest.raises(RuntimeError, match="step failed"):
        run(Env(fail_step=True), viewer=True, video_dir=tmp_path, num_video_episodes=1)
    assert events == ["video_closed"]
    assert viewer.closed
