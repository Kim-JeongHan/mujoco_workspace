"""Periodic training evaluation uses the live policy and preserves optimization."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import checkpoint_metadata
from mujoco_lab.learning.config.config import RolloutConfig, TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.evaluation import PolicyEvaluator
from mujoco_lab.learning.trainers.train_bc import run_training


class TinyEnv:
    physics_steps_per_action = 1
    simulator = SimpleNamespace(dt=0.002, data=SimpleNamespace(time=0.0))
    task = None

    observation_space = SimpleNamespace(shape=(1,))
    action_space = SimpleNamespace(
        shape=(1,), low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
    )

    @property
    def action_dt(self):
        return self.simulator.dt * self.physics_steps_per_action

    def reset(self, *, seed):
        self.simulator.data.time = 0.0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.simulator.data.time += self.simulator.dt
        return np.array([0.0], dtype=np.float32), 0.0, False, True, {"success": False}


@pytest.mark.parametrize("ratios", [(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (0.25, 0.25)])
def test_training_split_preserves_seeded_random_split_order(tmp_path, monkeypatch, ratios):
    from mujoco_lab.learning import train as cli

    config = TrainConfig(
        output_dir=tmp_path,
        validation_ratio=ratios[0],
        test_ratio=ratios[1],
        action_execution_hz=500,
        eval_interval=0,
        seed=7,
    )
    demonstrations = [
        Episode(
            states=np.zeros((2, 1), dtype=np.float32),
            actions=np.zeros((1, 1), dtype=np.float32),
            metadata={
                "seed": index,
                "replay": {
                    "physics_steps_per_action": 1,
                    "gripper_action_units": "opening_width_m",
                },
                "observation": {"frame_dim": 1, "rotation_indices": []},
            },
        )
        for index in range(8)
    ]
    captured = {}

    class CaptureLogger:
        def __init__(self, _path, settings, **kwargs):
            captured.update(settings["dataset"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def log(self, *args, **kwargs):
            pass

    def stop_before_training(*args, **kwargs):
        raise RuntimeError("split captured")

    monkeypatch.setattr(cli.tyro, "cli", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli, "load_episodes", lambda _path: demonstrations)
    monkeypatch.setattr(cli, "Logger", CaptureLogger)
    monkeypatch.setattr(cli, "run_training", stop_before_training)
    with pytest.raises(RuntimeError, match="split captured"):
        cli.main()
    validation_count = round(8 * ratios[0])
    test_count = round(8 * ratios[1])
    reference = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(torch.arange(8)),
        [8 - validation_count - test_count, validation_count, test_count],
        generator=torch.Generator().manual_seed(config.seed),
    )
    for name, subset in zip(("train", "validation", "test"), reference, strict=True):
        assert captured[f"{name}_seeds"] == list(subset.indices)


@pytest.mark.parametrize("ema_decay", [None, 0.5])
def test_evaluation_interval_final_step_and_training_rng(monkeypatch, ema_decay):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    demonstration = Episode(
        states=np.arange(6, dtype=np.float32).reshape(-1, 1),
        actions=np.arange(5, dtype=np.float32).reshape(-1, 1),
        metadata={
            "replay": {"physics_steps_per_action": 1},
            "observation": {"frame_dim": 1, "rotation_indices": []},
        },
    )
    demonstration.validate_training_data()
    config = TrainConfig(
        policy_type="flow",
        hidden_dims=(8,),
        obs_horizon=1,
        chunk_size=1,
        execution_horizon=1,
        action_execution_hz=500,
        batch_size=2,
        num_epochs=2,
        log_interval=100,
        eval_interval=4,
        ema_decay=ema_decay,
    )
    observed = []

    def evaluate(model, normalizer, step):
        assert model.training == (ema_decay is None)
        metadata = checkpoint_metadata(model, normalizer, config, optimizer_step=step)
        rows, summary = PolicyEvaluator(
            TinyEnv(),
            RolloutConfig(num_episodes=1, seed=100, max_seconds=0.002, video_episodes=0),
            torch.device("cpu"),
        ).evaluate(model, normalizer, metadata, flow_num_steps=2)
        assert model.training == (ema_decay is None)
        observed.append((step, rows[0]["env_seed"], summary["attempted"]))

    evaluated, _ = run_training(config, [demonstration], [], evaluate=evaluate)
    reference, _ = run_training(config, [demonstration])
    assert observed == [(4, 100, 1), (6, 100, 1)]
    for name, parameter in evaluated.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name])

    disabled = []
    run_training(
        replace(
            config,
            num_epochs=1,
            eval_interval=0,
            rollout=RolloutConfig(num_episodes=0, max_seconds=0),
        ),
        [demonstration],
        [],
        evaluate=lambda *_args: disabled.append(True),
    )
    assert disabled == []
