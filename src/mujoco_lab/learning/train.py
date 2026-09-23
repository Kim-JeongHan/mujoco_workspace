"""Train offline behavior cloning with local logs and optional W&B mirroring."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

import tyro

from mujoco_lab.learning.checkpoint import checkpoint_metadata, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets import load_episodes, split_episodes
from mujoco_lab.learning.datasets.loading import validate_episode_cadence
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.evaluation import (
    evaluate_policy,
    evaluation_log_metrics,
    validate_contract,
)
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.trainers.train_bc import run_training
from mujoco_lab.utils.logger import Logger as ConsoleLogger


def main() -> None:
    console = ConsoleLogger()
    config = tyro.cli(TrainConfig, description="Train offline cube-stack behavior cloning")
    config.validate()
    episodes = load_episodes(config.data_dir)
    validate_episode_cadence(episodes, config.physics_steps_per_action)
    train, validation, test = split_episodes(
        episodes,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    dataset_metadata = {
        "data_dir": str(config.data_dir),
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "test_episodes": len(test),
        "train_seeds": [episode.metadata.get("seed") for episode in train],
        "validation_seeds": [episode.metadata.get("seed") for episode in validation],
        "test_seeds": [episode.metadata.get("seed") for episode in test],
        "replay": train[0].metadata.get("replay", {}),
    }
    settings = asdict(config)
    settings["data_dir"] = str(config.data_dir)
    settings["output_dir"] = str(config.output_dir)
    settings["dataset"] = dataset_metadata
    eval_env = None
    eval_dt = None
    if config.eval_interval:
        eval_env, eval_dt, _ = create_evaluation_env(
            {"train_config": settings, "dataset_metadata": dataset_metadata},
            xy_range=config.eval_xy_range,
            min_gap=config.eval_min_gap,
            max_steps=config.eval_max_steps,
            cube_yaw_range_degrees=config.eval_cube_yaw_range_degrees,
        )
        if eval_env.observation_space.shape != train[0].states.shape[1:]:
            raise ValueError("Evaluation observation dimension differs from training episodes")
        if eval_env.action_space.shape != train[0].actions.shape[1:]:
            raise ValueError("Evaluation action dimension differs from training episodes")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.exp_name or f"{config.robot}-{config.policy_type}-seed{config.seed}"
    run_dir = (
        config.output_dir
        / "bc"
        / config.policy_type
        / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    )
    with Logger(
        run_dir,
        settings,
        run_name=run_name,
    ) as logger:
        logger.log(
            {
                "data/train_episodes": len(train),
                "data/validation_episodes": len(validation),
                "data/test_episodes": len(test),
            },
            step=0,
        )

        def evaluate(model, normalizer, step: int) -> None:
            assert eval_env is not None and eval_dt is not None
            metadata = checkpoint_metadata(
                model,
                normalizer,
                config,
                optimizer_step=step,
                dataset_metadata=dataset_metadata,
            )
            validate_contract(model, normalizer, metadata, eval_env)
            checkpoint = run_dir / f"checkpoint_step_{step:08d}.pt"
            save_checkpoint(
                checkpoint,
                model,
                normalizer,
                config,
                optimizer_step=step,
                dataset_metadata=dataset_metadata,
            )
            logger.log_checkpoint(checkpoint, step=step)
            step_dir = run_dir / "eval" / f"step_{step:08d}"
            step_dir.mkdir(parents=True, exist_ok=False)
            with (step_dir / "episodes.jsonl").open("x", encoding="utf-8") as episode_file:

                def record(row):
                    episode_file.write(json.dumps(row, allow_nan=False) + "\n")
                    episode_file.flush()
                    console.info(
                        f"Evaluation step={step} seed={row['env_seed']} "
                        f"success={row['success']} steps={row['steps']}"
                    )

                episodes, summary = evaluate_policy(
                    eval_env,
                    model,
                    normalizer,
                    metadata,
                    num_episodes=config.num_eval_episodes,
                    seed=config.eval_seed,
                    policy_seed=config.eval_policy_seed,
                    max_steps=config.eval_max_steps,
                    dt=eval_dt,
                    device=next(model.parameters()).device,
                    flow_num_steps=config.flow_num_steps,
                    on_episode=record,
                    video_dir=step_dir / "videos",
                    num_video_episodes=config.eval_video_episodes,
                    video_fps=config.eval_video_fps,
                    video_width=config.eval_video_width,
                    video_height=config.eval_video_height,
                )
            with (step_dir / "summary.json").open("x", encoding="utf-8") as summary_file:
                json.dump(summary, summary_file, indent=2, allow_nan=False)
                summary_file.write("\n")
            logger.log_evaluation(evaluation_log_metrics(summary), episodes, step=step)

        model, normalizer = run_training(
            config,
            train,
            validation,
            logger=logger,
            evaluate=evaluate if eval_env is not None else None,
        )
        train_chunks = sum(max(0, len(episode) - config.chunk_size + 1) for episode in train)
        optimizer_step = math.ceil(train_chunks / config.batch_size) * config.num_epochs
        checkpoint = run_dir / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model,
            normalizer,
            config,
            optimizer_step=optimizer_step,
            dataset_metadata=dataset_metadata,
        )
        logger.log_checkpoint(checkpoint, step=optimizer_step)
    console.info(f"Saved BC run to {run_dir}")


if __name__ == "__main__":
    main()
