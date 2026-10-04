"""Evaluate a saved BC checkpoint in the recorded physical task environment."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import tyro

from mujoco_lab.learning.checkpoint import load_checkpoint
from mujoco_lab.learning.config.config import EvalConfig
from mujoco_lab.learning.evaluation import (
    PolicyEvaluator,
    evaluation_log_metrics,
)
from mujoco_lab.learning.logging import Logger
from mujoco_lab.utils.logger import Logger as ConsoleLogger


def create_evaluation_env(
    metadata: dict[str, Any],
    *,
    xy_range: float | None,
    min_gap: float,
    max_steps: int,
    cube_yaw_range_degrees: float | None = None,
    book_yaw_range_degrees: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Choose the task builder using the scene recorded in the dataset."""
    scene = metadata["dataset_metadata"]["replay"].get("scene", "cube_stack")
    if scene == "book_insertion":
        from mujoco_lab.learning.evaluate_book import create_book_evaluation_env

        return create_book_evaluation_env(
            metadata,
            xy_range=xy_range,
            max_steps=max_steps,
            book_yaw_range_degrees=book_yaw_range_degrees,
        )
    if scene == "cube_stack":
        from mujoco_lab.learning.evaluate_cube import create_cube_evaluation_env

        return create_cube_evaluation_env(
            metadata,
            xy_range=xy_range,
            min_gap=min_gap,
            max_steps=max_steps,
            cube_yaw_range_degrees=cube_yaw_range_degrees,
        )
    raise ValueError(f"Unsupported evaluation scene: {scene}")


def run(config: EvalConfig, *, expected_scene: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Load, validate, evaluate, and write one exclusive local run directory."""
    console = ConsoleLogger()
    config.validate()
    rollout = config.rollout
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(config.device)
    model, normalizer, metadata = load_checkpoint(config.checkpoint)
    scene_name = metadata["dataset_metadata"]["replay"].get("scene", "cube_stack")
    if expected_scene is not None and scene_name != expected_scene:
        raise ValueError(f"Expected {expected_scene} checkpoint, recorded scene is {scene_name}")
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
        book_yaw_range_degrees=rollout.book_yaw_range_degrees,
    )
    model.to(device)

    eval_seeds = list(range(rollout.seed, rollout.seed + rollout.num_episodes))
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
        f"{scene['robot_name']}-{architecture['policy_type']}-eval-seed{rollout.seed}"
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

            evaluator = PolicyEvaluator(env, rollout, device)
            episodes, summary = evaluator.evaluate(
                model,
                normalizer,
                metadata,
                flow_num_steps=flow_num_steps,
                on_episode=record,
                video_dir=run_dir / "videos",
            )
        with (run_dir / "summary.json").open("x", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, allow_nan=False)
            summary_file.write("\n")
        logger.log_evaluation(
            evaluation_log_metrics(summary), episodes, step=metadata["optimizer_step"]
        )
    return run_dir, summary


def main() -> None:
    config = tyro.cli(EvalConfig, description="Evaluate a saved BC policy in its recorded scene")
    run_dir, summary = run(config)
    ConsoleLogger().info(f"Saved {summary['attempted']} evaluation attempts to {run_dir}")


if __name__ == "__main__":
    main()
