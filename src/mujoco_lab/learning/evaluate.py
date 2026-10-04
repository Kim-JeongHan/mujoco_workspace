"""Evaluate a saved BC checkpoint in the physical cube stacking environment."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import tyro

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import create_controller
from mujoco_lab.learning.checkpoint import load_checkpoint
from mujoco_lab.learning.config.config import EvalConfig
from mujoco_lab.learning.datasets.replay import cube_stack_metadata
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation import (
    evaluate_policy,
    evaluation_log_metrics,
)
from mujoco_lab.learning.logging import Logger
from mujoco_lab.tasks import CubeStackTask
from mujoco_lab.utils.logger import Logger as ConsoleLogger


def _verify_replay(simulator: Simulator, replay: dict[str, Any], *, robot: str, cubes: int) -> None:
    """Check the recorded scene against the current MuJoCo model."""
    actual = cube_stack_metadata(simulator, cubes=cubes, robot=robot)
    for name in ("model_sha256", "visual_sha256", "mujoco_version"):
        if replay[name] != actual[name]:
            raise ValueError(f"Checkpoint {name} differs from the current cube stack model")


def create_evaluation_env(
    metadata: dict[str, Any],
    *,
    xy_range: float,
    min_gap: float,
    max_steps: int,
    cube_yaw_range_degrees: float | None = None,
) -> tuple[CubeStackEnv, dict[str, Any]]:
    """Build the cube-stack environment from recorded scene metadata."""
    replay = metadata["dataset_metadata"]["replay"]
    action_repeat = (
        metadata["architecture"]["physics_steps_per_action"]
        if "architecture" in metadata
        else replay["physics_steps_per_action"]
    )
    robot_type = replay["robot"]
    robot_name = replay["robot_name"]
    cubes = replay["cubes"]
    dt = replay["dt"]
    yaw_range = (
        replay["cube_yaw_range_degrees"]
        if cube_yaw_range_degrees is None
        else cube_yaw_range_degrees
    )
    simulator = Simulator(
        create_cube_stack(cubes, environment=replay["environment"]),
        robots=[RobotSpec(robot_name, robot_type, config=load_robot_config(robot_type))],
        dt=dt,
    )
    _verify_replay(simulator, replay, robot=robot_type, cubes=cubes)
    robot = simulator.robots[robot_name]
    controller = (
        create_controller(
            robot,
            replace(
                load_robot_config(robot.robot_type).controller,
                name="position",
                gravity_compensation=True,
                frame="grasp",
            ),
        )
        if robot_type == "panda"
        else create_controller(
            robot, replace(load_robot_config(robot.robot_type).controller, name="pd", frame="grasp")
        )
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
        "environment": replay["environment"],
        "dt": simulator.dt,
        "physics_steps_per_action": action_repeat,
        "cube_yaw_range_degrees": yaw_range,
        "action_dt": env.action_dt,
    }
    return env, scene


def run(config: EvalConfig) -> tuple[Path, dict[str, Any]]:
    """Load, validate, evaluate, and write one exclusive local run directory."""
    console = ConsoleLogger()
    config.validate()
    rollout = config.rollout
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(config.device)
    model, normalizer, metadata = load_checkpoint(config.checkpoint)
    architecture = metadata["architecture"]
    flow_num_steps = metadata["train_config"]["flow_num_steps"]
    if not isinstance(flow_num_steps, int) or flow_num_steps <= 0:
        raise ValueError("flow_num_steps must be a positive integer")
    env, scene = create_evaluation_env(
        metadata,
        xy_range=rollout.xy_range,
        min_gap=rollout.min_gap,
        max_steps=rollout.max_steps,
        cube_yaw_range_degrees=rollout.cube_yaw_range_degrees,
    )
    model.to(device)

    eval_seeds = list(range(rollout.env_seed, rollout.env_seed + rollout.num_episodes))
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
        "policy_seeds": list(
            range(rollout.policy_seed, rollout.policy_seed + rollout.num_episodes)
        ),
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
        f"{scene['robot_name']}-{architecture['policy_type']}-eval-seed{rollout.env_seed}"
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
                num_episodes=rollout.num_episodes,
                seed=rollout.env_seed,
                policy_seed=rollout.policy_seed,
                max_steps=rollout.max_steps,
                device=device,
                flow_num_steps=flow_num_steps,
                on_episode=record,
                video_dir=run_dir / "videos",
                num_video_episodes=rollout.video_episodes,
                video_fps=rollout.video_fps,
                video_width=rollout.video_width,
                video_height=rollout.video_height,
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
