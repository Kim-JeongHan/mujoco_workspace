"""Check training and evaluation invariants where their settings are consumed."""

from dataclasses import replace

import pytest
import tyro

from mujoco_lab.learning.collect_book import Config as BookConfig
from mujoco_lab.learning.collect_cube import Config as CubeConfig
from mujoco_lab.learning.config import BCTrainConfig
from mujoco_lab.learning.timing import max_steps_for_seconds
from mujoco_lab.learning.trainers.train_bc import run_training
from mujoco_lab.planning import RRTStarConfig, planner_name


def test_cube_collection_cli_selects_rrt_star_and_continued_optimization():
    config = tyro.cli(
        CubeConfig,
        args=[
            "--method",
            "sampling",
            "--cubes",
            "4",
            "--count",
            "1000",
            "planning:rrt-star-config",
            "--planning.max-iterations",
            "1000",
            "--planning.step-size",
            "0.2",
            "--planning.goal-tolerance",
            "0.04",
            "--planning.seed",
            "7",
            "--planning.no-return-first-solution",
        ],
    )
    assert config.cubes == 4 and config.count == 1000 and config.method == "sampling"
    assert isinstance(config.planning, RRTStarConfig)
    assert planner_name(config.planning) == "rrt_star"
    assert config.planning.max_iterations == 1000
    assert config.planning.seed == 7 and not config.planning.return_first_solution


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"execution_horizon": 0}, "execution_horizon"),
        ({"execution_horizon": 17}, "execution_horizon"),
        ({"num_epochs": 0}, "num_epochs"),
        ({"sampling_num_steps": 0}, "sampling_num_steps"),
        (
            {"policy_type": "diffusion", "sampling_num_steps": 101},
            "Diffusion sampling steps",
        ),
    ],
)
def test_training_rejects_project_invariants(overrides, field):
    with pytest.raises(ValueError, match=field):
        run_training(replace(BCTrainConfig(), **overrides), [])


@pytest.mark.parametrize("config_type", [BCTrainConfig, CubeConfig, BookConfig])
@pytest.mark.parametrize("simulation_hz", [500.00000001])
def test_cadence_rounds_accepted_float_ratios(config_type, simulation_hz):
    config = config_type(simulation_hz=simulation_hz)
    assert config.physics_steps_per_action == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"simulation_hz": 0},
        {"action_execution_hz": 120},
    ],
)
def test_cadence_rejects_invalid_frequencies_without_explicit_validation(overrides):
    with pytest.raises(ValueError, match="simulation_hz"):
        _ = BCTrainConfig(**overrides).physics_steps_per_action


@pytest.mark.parametrize(
    "ratios",
    [(-0.1, 0), (float("nan"), 0), (0.6, 0.4)],
)
def test_training_split_rejects_invalid_ratios(ratios, monkeypatch):
    from mujoco_lab.learning import train as cli

    config = BCTrainConfig(validation_ratio=ratios[0], test_ratio=ratios[1])
    monkeypatch.setattr(cli.tyro, "cli", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(cli, "load_episodes", lambda _path: pytest.fail("Invalid holdout accepted"))
    with pytest.raises(ValueError, match="Holdout ratios"):
        cli.main()


@pytest.mark.parametrize("action_execution_hz", [50.0, 100.0])
def test_collection_duration_overrides_cover_the_requested_time(action_execution_hz):
    for config_type in (CubeConfig, BookConfig):
        config = config_type(max_seconds=0.025, action_execution_hz=action_execution_hz)
        elapsed = config.max_steps / action_execution_hz
        assert 0.025 <= elapsed < 0.025 + 1.0 / action_execution_hz


@pytest.mark.parametrize("config_type", [CubeConfig, BookConfig])
@pytest.mark.parametrize("max_seconds", [0.0])
def test_collection_rejects_invalid_durations(config_type, max_seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        _ = config_type(max_seconds=max_seconds).max_steps


@pytest.mark.parametrize("action_dt", [0.0])
def test_duration_conversion_rejects_invalid_action_intervals(action_dt):
    with pytest.raises(ValueError, match="action_dt"):
        max_steps_for_seconds(15.0, action_dt)
