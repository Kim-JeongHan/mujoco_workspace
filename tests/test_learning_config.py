"""Check training and evaluation invariants where their settings are consumed."""

from dataclasses import replace
from typing import Literal

import pytest

from mujoco_lab.learning.collect_book import Config as BookConfig
from mujoco_lab.learning.collect_cube import Config as CubeConfig
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.timing import max_steps_for_seconds
from mujoco_lab.learning.trainers.train_bc import run_training


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"execution_horizon": 0}, "execution_horizon"),
        ({"execution_horizon": 17}, "execution_horizon"),
        ({"num_epochs": 0}, "num_epochs"),
    ],
)
def test_training_rejects_project_invariants(overrides, field):
    with pytest.raises(ValueError, match=field):
        run_training(replace(TrainConfig(), **overrides), [])


@pytest.mark.parametrize("config_type", [TrainConfig, CubeConfig, BookConfig])
@pytest.mark.parametrize("simulation_hz", [500.0, 499.99999999, 500.00000001])
def test_cadence_rounds_accepted_float_ratios(config_type, simulation_hz):
    config = config_type(simulation_hz=simulation_hz)
    assert config.physics_steps_per_action == 5


@pytest.mark.parametrize("config_type", [TrainConfig, CubeConfig, BookConfig])
@pytest.mark.parametrize(
    "overrides",
    [
        {"simulation_hz": 0},
        {"simulation_hz": -500},
        {"simulation_hz": float("inf")},
        {"action_execution_hz": 0},
        {"action_execution_hz": float("nan")},
        {"action_execution_hz": 120},
        {"action_execution_hz": 1000},
    ],
)
def test_cadence_rejects_invalid_frequencies_without_explicit_validation(config_type, overrides):
    with pytest.raises(ValueError, match="simulation_hz"):
        _ = config_type(**overrides).physics_steps_per_action


@pytest.mark.parametrize(
    "ratios",
    [(-0.1, 0), (float("nan"), 0), (0.6, 0.4)],
)
def test_training_split_rejects_invalid_ratios(ratios, monkeypatch):
    from mujoco_lab.learning import train as cli

    config = TrainConfig(validation_ratio=ratios[0], test_ratio=ratios[1])
    monkeypatch.setattr(cli.tyro, "cli", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(cli, "load_episodes", lambda _path: pytest.fail("Invalid holdout accepted"))
    with pytest.raises(ValueError, match="Holdout ratios"):
        cli.main()


@pytest.mark.parametrize("action_execution_hz", [50.0, 100.0])
@pytest.mark.parametrize(("method", "duration"), [("heuristic", 60.0), ("sampling", 180.0)])
def test_collection_preserves_default_durations_and_accepts_overrides(
    action_execution_hz, method: Literal["heuristic", "sampling"], duration: float
):
    config = CubeConfig(method=method, action_execution_hz=action_execution_hz)
    assert config.max_steps == duration * action_execution_hz
    overridden = replace(config, max_seconds=0.025)
    elapsed = overridden.max_steps / action_execution_hz
    assert 0.025 <= elapsed < 0.025 + 1.0 / action_execution_hz
    book = BookConfig(action_execution_hz=action_execution_hz)
    assert book.max_steps == 180.0 * action_execution_hz


@pytest.mark.parametrize("config_type", [CubeConfig, BookConfig])
@pytest.mark.parametrize("max_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_collection_rejects_invalid_durations(config_type, max_seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        _ = config_type(max_seconds=max_seconds).max_steps


@pytest.mark.parametrize("action_dt", [0.0, -0.01, float("inf"), float("nan")])
def test_duration_conversion_rejects_invalid_action_intervals(action_dt):
    with pytest.raises(ValueError, match="action_dt"):
        max_steps_for_seconds(15.0, action_dt)
