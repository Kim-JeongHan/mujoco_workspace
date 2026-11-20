"""Check training and evaluation invariants where their settings are consumed."""

import pytest
import tyro

from mujoco_lab.learning.collect_book import Config as BookConfig
from mujoco_lab.learning.collect_cube import Config as CubeConfig
from mujoco_lab.learning.config import (
    BCTrainConfig,
    DiffusionBCTrainConfig,
    FlowBCTrainConfig,
    MSEBCTrainConfig,
)
from mujoco_lab.learning.policies import DiffusionPolicy, build_policy
from mujoco_lab.learning.timing import max_steps_for_seconds
from mujoco_lab.learning.train import parse_train_config
from mujoco_lab.planning import RRTStarConfig, planner_name


@pytest.mark.parametrize(
    "policy_type,config_type,args",
    [
        ("mse", MSEBCTrainConfig, ["--hidden-dims", "8", "16"]),
        (
            "flow",
            FlowBCTrainConfig,
            ["--hidden-dims", "8", "16", "--time-embed-dim", "64"],
        ),
        (
            "diffusion",
            DiffusionBCTrainConfig,
            [
                "--down-dims",
                "16",
                "32",
                "--time-embed-dim",
                "16",
                "--scheduler.num-train-timesteps",
                "40",
                "--scheduler.beta-schedule",
                "linear",
                "--scheduler.timestep-spacing",
                "trailing",
                "--sampling-num-steps",
                "7",
            ],
        ),
    ],
)
def test_bc_cli_reads_policy_specific_settings(policy_type, config_type, args):
    config = parse_train_config(["--policy-type", policy_type, *args, "--chunk-size", "3"])
    assert isinstance(config, config_type)
    assert config.policy_type == policy_type
    model = build_policy(config, state_dim=4, action_dim=2)
    assert model.chunk_size == 3
    if isinstance(model, DiffusionPolicy):
        assert model.denoiser.down_dims == (16, 32)
        assert model.denoiser.time_embed_dim == 16
        assert model.scheduler.config["num_train_timesteps"] == 40
        assert model.scheduler.config["beta_schedule"] == "linear"
        assert model.scheduler.config["timestep_spacing"] == "trailing"
    else:
        assert model.net[0].out_features == 8
        if isinstance(config, FlowBCTrainConfig):
            assert model.time_embed_dim == 64


def test_flow_cli_rejects_diffusion_scheduler_options():
    with pytest.raises(SystemExit):
        parse_train_config(["--policy-type", "flow", "--scheduler.num-train-timesteps", "40"])


@pytest.mark.parametrize(
    "args,config_type",
    [
        ([], MSEBCTrainConfig),
        (["--policy-type=flow"], FlowBCTrainConfig),
        (["--policy_type", "diffusion"], DiffusionBCTrainConfig),
    ],
)
def test_training_selects_config_from_common_policy_type(args, config_type):
    assert isinstance(parse_train_config(args), config_type)


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

    config = MSEBCTrainConfig(validation_ratio=ratios[0], test_ratio=ratios[1])
    monkeypatch.setattr(cli.tyro, "cli", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        cli.Episode, "load_all", lambda _path: pytest.fail("Invalid holdout accepted")
    )
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
