"""Evaluate a saved BC checkpoint in the physical cube stacking environment."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import tyro

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.control import create_controller
from mujoco_lab.learning.checkpoint import checkpoint_action_repeat, load_checkpoint
from mujoco_lab.learning.config.config import EvalConfig
from mujoco_lab.learning.datasets.replay import cube_stack_metadata, replay_cube_yaw_range_degrees
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation import (
    evaluate_policy,
    evaluation_log_metrics,
    validate_contract,
)
from mujoco_lab.learning.logging import Logger
from mujoco_lab.tasks import CubeStackTask
from mujoco_lab.utils.logger import Logger as ConsoleLogger


def _scene_settings(metadata: dict[str, Any]) -> tuple[str, str, int, str, float, dict[str, str]]:
    """Prefer recorded replay settings and identify each fallback in the run config."""
    replay = metadata.get("dataset_metadata", {}).get("replay") or {}
    training = metadata.get("train_config") or {}
    settings: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key in ("robot", "cubes", "environment", "dt"):
        replay_value = replay.get(key)
        train_value = training.get(key)
        if replay_value is not None:
            if train_value is not None and train_value != replay_value:
                raise ValueError(f"Checkpoint replay {key} differs from train_config {key}")
            settings[key] = replay_value
            sources[key] = "dataset_metadata.replay"
        elif train_value is not None:
            settings[key] = train_value
            sources[key] = "train_config"
        elif key == "dt" and metadata.get("dataset_metadata", {}).get("dt") is not None:
            settings[key] = metadata["dataset_metadata"]["dt"]
            sources[key] = "dataset_metadata"
        else:
            raise ValueError(f"Checkpoint lacks essential scene setting {key}")
    if replay and replay.get("scene", "cube_stack") != "cube_stack":
        raise ValueError("Checkpoint replay scene is not cube_stack")
    if settings["robot"] not in ("panda", "forte"):
        raise ValueError(f"Unsupported checkpoint robot: {settings['robot']}")
    if not isinstance(settings["cubes"], int) or settings["cubes"] <= 0:
        raise ValueError("Checkpoint cubes must be a positive integer")
    if settings["environment"] not in ("table_shelf", "warehouse"):
        raise ValueError(f"Unsupported checkpoint environment: {settings['environment']}")
    dt = settings["dt"]
    if not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Checkpoint dt must be finite and positive")
    robot_name = replay.get("robot_name", settings["robot"])
    if not isinstance(robot_name, str) or not robot_name or "/" in robot_name:
        raise ValueError("Checkpoint robot_name must be a nonempty instance name")
    sources["robot_name"] = "dataset_metadata.replay" if "robot_name" in replay else "robot"
    return (
        settings["robot"],
        robot_name,
        settings["cubes"],
        settings["environment"],
        float(dt),
        sources,
    )


def _verify_replay(
    simulator: Simulator, metadata: dict[str, Any], *, robot: str, cubes: int, dt: float
) -> None:
    """Check scene geometry and timebase before choosing a robot controller."""
    replay = metadata.get("dataset_metadata", {}).get("replay") or {}
    actual = cube_stack_metadata(simulator, cubes=cubes, robot=robot)
    if not np.isclose(simulator.dt, dt, rtol=0, atol=1e-12):
        raise ValueError(f"Checkpoint dt={dt} differs from simulator dt={simulator.dt}")
    if replay:
        schema = replay.get("schema_version", 1)
        signature = "visual_sha256" if schema == 2 else "model_sha256" if schema == 1 else None
        if signature is None:
            raise ValueError(f"Unsupported checkpoint replay schema_version={schema}")
        if not replay.get(signature):
            raise ValueError(f"Checkpoint replay lacks {signature}")
        for name in ("model_sha256", "visual_sha256"):
            if name in replay and replay[name] != actual[name]:
                raise ValueError(f"Checkpoint {name} differs from the current cube stack model")
        if replay.get("mujoco_version") and replay["mujoco_version"] != actual["mujoco_version"]:
            raise ValueError("Checkpoint MuJoCo version differs from the current version")


def create_evaluation_env(
    metadata: dict[str, Any],
    *,
    xy_range: float,
    min_gap: float,
    max_steps: int,
    cube_yaw_range_degrees: float | None = None,
) -> tuple[CubeStackEnv, float, dict[str, Any]]:
    """Build the recorded scene; returned dt is one physics tick, not one action."""
    action_repeat = checkpoint_action_repeat(metadata)
    replay = metadata.get("dataset_metadata", {}).get("replay") or {}
    yaw_range = (
        replay_cube_yaw_range_degrees(replay)
        if cube_yaw_range_degrees is None
        else replay_cube_yaw_range_degrees({"cube_yaw_range_degrees": cube_yaw_range_degrees})
    )
    robot_type, robot_name, cubes, environment, dt, sources = _scene_settings(metadata)
    simulator = Simulator(
        create_cube_stack(cubes, environment=environment),
        robots=[RobotSpec(robot_name, robot_type)],
        dt=dt,
    )
    _verify_replay(simulator, metadata, robot=robot_type, cubes=cubes, dt=dt)
    robot = simulator.robots[robot_name]
    controller = (
        create_controller("position", robot, gravity_compensation=True, frame="grasp")
        if robot_type == "panda"
        else create_controller("pd", robot, frame="grasp")
    )
    robot.change_controller(controller)
    task = CubeStackTask(simulator, cubes)
    env = CubeStackEnv(
        task,
        xy_range=xy_range,
        min_gap=min_gap,
        cube_yaw_range_degrees=yaw_range,
        max_steps=max_steps,
        physics_steps_per_action=action_repeat,
    )
    scene = {
        "robot": robot_type,
        "robot_name": robot_name,
        "cubes": cubes,
        "environment": environment,
        "dt": dt,
        "physics_steps_per_action": action_repeat,
        "cube_yaw_range_degrees": yaw_range,
        "action_dt": dt * action_repeat,
        "scene_setting_sources": sources,
    }
    return env, dt, scene


def run(config: EvalConfig) -> tuple[Path, dict[str, Any]]:
    """Load, validate, evaluate, and write one exclusive local run directory."""
    console = ConsoleLogger()
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(config.device)
    model, normalizer, metadata = load_checkpoint(config.checkpoint)
    architecture = metadata["architecture"]
    if architecture["policy_type"] not in ("mse", "flow"):
        raise ValueError("Only MSE and Flow checkpoints are supported for evaluation")
    flow_num_steps = metadata["train_config"].get("flow_num_steps", 10)
    if not isinstance(flow_num_steps, int) or flow_num_steps <= 0:
        raise ValueError("flow_num_steps must be a positive integer")
    env, dt, scene = create_evaluation_env(
        metadata,
        xy_range=config.xy_range,
        min_gap=config.min_gap,
        max_steps=config.max_steps,
        cube_yaw_range_degrees=config.cube_yaw_range_degrees,
    )
    for key, source in scene["scene_setting_sources"].items():
        if source != "dataset_metadata.replay":
            console.warn(f"Checkpoint {key} comes from {source}")
    validate_contract(model, normalizer, metadata, env)
    model.to(device)

    eval_seeds = list(range(config.seed, config.seed + config.num_episodes))
    dataset = metadata.get("dataset_metadata", {})
    overlaps = {
        split: sorted(set(eval_seeds) & set(dataset.get(f"{split}_seeds", [])))
        for split in ("train", "validation", "test")
    }
    for split, seeds in overlaps.items():
        if seeds:
            console.warn(f"Evaluation seeds overlap {split} seeds: {seeds}")
    settings = asdict(config)
    settings["checkpoint"] = str(config.checkpoint.resolve())
    settings["output_dir"] = str(config.output_dir)
    settings["effective"] = {
        "device": str(device),
        "flow_num_steps": flow_num_steps,
        **scene,
        "env_seeds": eval_seeds,
        "policy_seeds": list(range(config.policy_seed, config.policy_seed + config.num_episodes)),
        "seed_overlaps": overlaps,
        "checkpoint_metadata": metadata,
    }
    run_dir = (
        config.output_dir
        / "eval"
        / architecture["policy_type"]
        / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    )
    name = config.wandb_name or (
        f"{scene['robot_name']}-{architecture['policy_type']}-eval-seed{config.seed}"
    )
    with Logger(
        run_dir,
        settings,
        run_name=name,
    ) as logger:
        with (run_dir / "episodes.jsonl").open("x", encoding="utf-8") as episode_file:

            def record(row: dict[str, Any]) -> None:
                episode_file.write(json.dumps(row, allow_nan=False) + "\n")
                episode_file.flush()
                console.info(
                    f"Episode {row['env_seed']}: success={row['success']} "
                    f"steps={row['steps']} reason={row['termination_reason']}"
                )

            episodes, summary = evaluate_policy(
                env,
                model,
                normalizer,
                metadata,
                num_episodes=config.num_episodes,
                seed=config.seed,
                policy_seed=config.policy_seed,
                max_steps=config.max_steps,
                dt=dt,
                device=device,
                flow_num_steps=flow_num_steps,
                on_episode=record,
                video_dir=run_dir / "videos",
                num_video_episodes=config.num_video_episodes,
                video_fps=config.video_fps,
                video_width=config.video_width,
                video_height=config.video_height,
            )
        with (run_dir / "summary.json").open("x", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, allow_nan=False)
            summary_file.write("\n")
        logger.log_evaluation(
            evaluation_log_metrics(summary), episodes, step=metadata["optimizer_step"]
        )
    return run_dir, summary


def main() -> None:
    config = tyro.cli(EvalConfig, description="Evaluate a saved cube-stack BC policy")
    run_dir, summary = run(config)
    ConsoleLogger().info(f"Saved {summary['attempted']} evaluation attempts to {run_dir}")


if __name__ == "__main__":
    main()
