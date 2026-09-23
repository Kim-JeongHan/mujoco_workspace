"""Training and evaluation reject invalid project settings before external work."""

from dataclasses import replace

import numpy as np
import pytest

from mujoco_lab.learning.config.config import EvalConfig, TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.evaluate import run
from mujoco_lab.learning.trainers.train_bc import run_training


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"obs_horizon": 0}, "obs_horizon"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"execution_horizon": 17}, "execution_horizon"),
        ({"physics_steps_per_action": 0}, "physics_steps_per_action"),
        ({"num_epochs": 0}, "num_epochs"),
        ({"log_interval": 0}, "log_interval"),
        ({"eval_interval": -1}, "eval_interval"),
        ({"num_eval_episodes": 0}, "num_eval_episodes"),
        ({"eval_max_steps": 0}, "eval_max_steps"),
        ({"eval_video_episodes": 2}, "eval_video_episodes"),
    ],
)
def test_train_config_rejects_project_invariants(overrides, field):
    with pytest.raises(ValueError, match=field):
        replace(TrainConfig(), **overrides).validate()


def test_disabled_training_evaluation_ignores_unused_settings():
    replace(
        TrainConfig(),
        eval_interval=0,
        num_eval_episodes=0,
        eval_max_steps=0,
        eval_video_episodes=-1,
        eval_video_width=1,
    ).validate()


def test_direct_training_validates_before_episode_access():
    with pytest.raises(ValueError, match="num_epochs"):
        run_training(TrainConfig(num_epochs=0), [])


def test_batch_size_reaches_dataloader_validation():
    episode = Episode(
        states=np.zeros((3, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
    )
    config = TrainConfig(
        batch_size=0,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        eval_interval=0,
    )
    with pytest.raises(ValueError, match="batch_size should be a positive integer"):
        run_training(config, [episode])


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"num_episodes": 0}, "num_episodes"),
        ({"max_steps": 0}, "max_steps"),
        ({"num_video_episodes": 4}, "num_video_episodes"),
    ],
)
def test_eval_config_rejects_project_invariants_before_checkpoint_load(
    tmp_path, monkeypatch, overrides, field
):
    def unexpected_load(_path):
        pytest.fail("Checkpoint must not load before config validation")

    monkeypatch.setattr("mujoco_lab.learning.evaluate.load_checkpoint", unexpected_load)
    config = replace(EvalConfig(checkpoint=tmp_path / "missing.pt"), **overrides)
    with pytest.raises(ValueError, match=field):
        run(config)
