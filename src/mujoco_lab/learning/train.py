"""Train offline behavior cloning with local logs and optional W&B mirroring."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

import torch
import tyro

from mujoco_lab.learning.checkpoint import checkpoint_metadata, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets import load_episodes
from mujoco_lab.learning.datasets.replay import require_width_actions
from mujoco_lab.learning.evaluate import create_rollout_env
from mujoco_lab.learning.evaluation import (
    PolicyEvaluator,
    evaluation_log_metrics,
)
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.trainers.train_bc import run_training
from mujoco_lab.utils.logger import Logger as ConsoleLogger


def main() -> None:
    console = ConsoleLogger()
    config = tyro.cli(TrainConfig, description="Train offline behavior cloning")
    repeat = config.physics_steps_per_action
    if not (
        config.validation_ratio >= 0
        and config.test_ratio >= 0
        and config.validation_ratio + config.test_ratio < 1
    ):
        raise ValueError("Holdout ratios must be nonnegative and sum to less than 1")
    episodes = load_episodes(config.data_dir)
    for episode in episodes:
        require_width_actions(episode.metadata.get("replay", {}))
        episode.check_physics_step_consistency(repeat, simulation_dt=config.simulation_dt)
    validation_count = (
        max(1, round(len(episodes) * config.validation_ratio)) if config.validation_ratio else 0
    )
    test_count = max(1, round(len(episodes) * config.test_ratio)) if config.test_ratio else 0
    train_count = len(episodes) - validation_count - test_count
    if train_count <= 0:
        raise ValueError("Not enough episodes for the requested holdouts and train")
    indices = torch.randperm(
        len(episodes), generator=torch.Generator().manual_seed(config.seed)
    ).tolist()
    validation_end = train_count + validation_count
    train = [episodes[index] for index in indices[:train_count]]
    validation = [episodes[index] for index in indices[train_count:validation_end]]
    test = [episodes[index] for index in indices[validation_end:]]
    dataset_metadata = {
        "data_dir": str(config.data_dir),
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "test_episodes": len(test),
        "train_seeds": [episode.metadata.get("seed") for episode in train],
        "validation_seeds": [episode.metadata.get("seed") for episode in validation],
        "test_seeds": [episode.metadata.get("seed") for episode in test],
        "replay": train[0].metadata["replay"],
    }
    if "observation" in train[0].metadata:
        dataset_metadata["observation"] = train[0].metadata["observation"]
    settings = asdict(config)
    settings["data_dir"] = str(config.data_dir)
    settings["output_dir"] = str(config.output_dir)
    settings["dataset"] = dataset_metadata
    rollout = config.rollout
    eval_env = None
    if config.eval_interval > 0:
        eval_env, _ = create_rollout_env(
            {"train_config": settings, "dataset_metadata": dataset_metadata},
            rollout,
        )
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

        evaluator = None

        def evaluate(model, normalizer, step: int) -> None:
            nonlocal evaluator
            assert eval_env is not None
            if evaluator is None:
                evaluator = PolicyEvaluator(eval_env, rollout, next(model.parameters()).device)
            metadata = checkpoint_metadata(
                model,
                normalizer,
                config,
                optimizer_step=step,
                dataset_metadata=dataset_metadata,
            )
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

                episodes, summary = evaluator.evaluate(
                    model,
                    normalizer,
                    metadata,
                    flow_num_steps=config.flow_num_steps,
                    on_episode=record,
                    video_dir=step_dir / "videos",
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
