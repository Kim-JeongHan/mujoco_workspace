"""EMA training keeps optimization and saved inference weights distinct."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.sequence import ChunkDataset
from mujoco_lab.learning.trainers.train_bc import run_training


def test_ema_updates_after_optimizer_and_returns_plain_checkpoint_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = TrainConfig(
        policy_type="mse",
        obs_horizon=1,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        hidden_dims=(),
        batch_size=1,
        num_epochs=1,
        lr=0.05,
        weight_decay=0,
        ema_decay=0.5,
        eval_interval=1,
        log_interval=100,
    )
    episode = Episode(
        states=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        actions=np.array([[0.0], [2.0], [4.0]], dtype=np.float32),
        metadata={"replay": {"physics_steps_per_action": 1}},
    )
    raw_history = []
    raw_parameter_ids = set()
    original_step = torch.optim.AdamW.step

    def record_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        assert all(p.requires_grad for p in parameters)
        raw_parameter_ids.update(id(p) for p in parameters)
        raw_history.append([p.detach().clone() for p in parameters])
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    evaluated = []
    log_rows = []

    class CaptureLogger:
        def log(self, values, *, step):
            log_rows.append((step, values))

    def evaluate(model, normalizer, step):
        assert not model.training
        parameters = list(model.parameters())
        assert all(not p.requires_grad and p.grad is None for p in parameters)
        assert all(id(p) not in raw_parameter_ids for p in parameters)
        expected = raw_history[0]
        for snapshot in raw_history[1:]:
            expected = [0.5 * old + 0.5 * new for old, new in zip(expected, snapshot, strict=True)]
        for actual, desired in zip(parameters, expected, strict=True):
            torch.testing.assert_close(actual, desired)
        evaluated.append((model, step))

    model, normalizer = run_training(
        config, [episode], [episode], logger=CaptureLogger(), evaluate=evaluate
    )
    assert [step for _, step in evaluated] == [1, 2, 3]
    assert all(candidate is model for candidate, _ in evaluated)
    assert not any(key.startswith("module.") for key in model.state_dict())
    validation = ChunkDataset([episode], 1, normalizer, obs_horizon=1)
    with torch.inference_mode():
        expected_loss = np.mean(
            [
                model.compute_loss(
                    torch.from_numpy(state[None]), torch.from_numpy(action[None])
                ).item()
                for state, action in validation
            ]
        )
    logged_loss = next(
        values["validation/loss"] for _, values in log_rows if "validation/loss" in values
    )
    assert logged_loss == pytest.approx(expected_loss)
    path = tmp_path / "ema.pt"
    save_checkpoint(path, model, normalizer, config, optimizer_step=3)
    loaded, _, metadata = load_checkpoint(path)
    assert metadata["optimizer_step"] == 3
    for actual, desired in zip(loaded.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, desired)


@pytest.mark.parametrize("ema_decay", [-0.1, 1.0, float("nan")])
def test_invalid_ema_decay_fails_before_loading_episodes(ema_decay):
    with pytest.raises(ValueError, match="ema_decay"):
        run_training(TrainConfig(ema_decay=ema_decay), [])


def test_ema_disabled_returns_trainable_raw_policy(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = TrainConfig(
        policy_type="mse",
        obs_horizon=1,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        hidden_dims=(),
        batch_size=1,
        num_epochs=1,
        ema_decay=None,
        eval_interval=0,
        log_interval=100,
    )
    episode = Episode(
        states=np.zeros((3, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        metadata={"replay": {"physics_steps_per_action": 1}},
    )
    model, _ = run_training(config, [episode])
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
