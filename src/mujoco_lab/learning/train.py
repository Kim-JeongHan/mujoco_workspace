"""Train offline behavior cloning with local logs and optional W&B mirroring."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

import torch
import tyro

from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.learning.checkpoint import config_dict, save_checkpoint
from mujoco_lab.learning.config import (
    BCPolicyTrainConfig,
    BCTrainConfig,
    DiffusionBCTrainConfig,
    FlowBCTrainConfig,
    MSEBCTrainConfig,
)
from mujoco_lab.learning.datasets import Episode
from mujoco_lab.learning.evaluate import create_rollout_env
from mujoco_lab.learning.evaluation import (
    PolicyEvaluator,
    evaluation_log_metrics,
)
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.metadata import DatasetMetadata, describe_episodes
from mujoco_lab.learning.trainers.train_bc import run_training
from mujoco_lab.utils.logger import Logger as ConsoleLogger


def parse_train_config(args: Sequence[str] | None = None) -> BCPolicyTrainConfig:
    """Select the policy settings before parsing the full training command."""
    config_types = {
        "mse": MSEBCTrainConfig,
        "flow": FlowBCTrainConfig,
        "diffusion": DiffusionBCTrainConfig,
    }
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument(
        "--policy-type", "--policy_type", choices=config_types, default=BCTrainConfig().policy_type
    )
    selected, _ = selector.parse_known_args(args)
    config_type = config_types[selected.policy_type]
    return tyro.cli(
        config_type,
        args=args,
        default=config_type(policy_type=selected.policy_type),
        description="Train offline behavior cloning",
    )


def main() -> None:
    console = ConsoleLogger()
    config = parse_train_config()
    sampling_num_steps = (
        config.sampling_num_steps
        if isinstance(config, (FlowBCTrainConfig, DiffusionBCTrainConfig))
        else 1
    )
    repeat = config.physics_steps_per_action
    if not (
        config.validation_ratio >= 0
        and config.test_ratio >= 0
        and config.validation_ratio + config.test_ratio < 1
    ):
        raise ValueError("Holdout ratios must be nonnegative and sum to less than 1")
    episodes = Episode.load_all(config.data_dir)
    for episode in episodes:
        episode.check_physics_step_consistency(repeat, simulation_dt=config.simulation_dt)
        # Playback arrays are not used by BC or policy evaluation; release their memory.
        episode.qpos = None
        episode.frame_times = None
        episode.mocap_pos = None
        episode.mocap_quat = None
    environment, observation = describe_episodes(episodes)
    if config.split_file is not None:
        split = json.loads(config.split_file.read_text())
        by_seed = {episode.metadata["seed"]: episode for episode in episodes}
        seed_lists = [split.get(f"{name}_seeds", []) for name in ("train", "validation", "test")]
        assigned = [seed for seeds in seed_lists for seed in seeds]
        if len(set(assigned)) != len(assigned):
            raise ValueError("Split seed assignments must be unique and disjoint")
        train, validation, test = [
            [by_seed[seed] for seed in seeds if seed in by_seed] for seeds in seed_lists
        ]
        if not train:
            raise ValueError("The split file selects no successful training episodes")
    else:
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
    dataset = DatasetMetadata.from_splits(config.data_dir, train, validation, test)
    settings = config_dict(config)
    robot_config = load_robot_config(environment.scene.robot)
    settings["controller"] = asdict(robot_config.controller)
    settings["reference"] = asdict(robot_config.reference)
    if robot_config.reference.name == "position_hold":
        settings["desired_velocity"] = "zero"
    elif robot_config.controller.name == "pd":
        settings["desired_velocity"] = robot_config.controller.desired_velocity
    rollout = config.evaluation
    eval_env = None
    if config.eval_interval_steps > 0:
        eval_env, _ = create_rollout_env(environment, rollout)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.run_name or f"{config.robot}-{config.policy_type}-seed{config.seed}"
    run_dir = (
        config.output_dir
        / "bc"
        / config.policy_type
        / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    )
    with Logger(
        run_dir,
        {**settings, "dataset": asdict(dataset), "environment": asdict(environment)},
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
            checkpoint = run_dir / f"checkpoint_step_{step:08d}.pt"
            metadata = save_checkpoint(
                checkpoint,
                model,
                normalizer,
                config,
                optimizer_step=step,
                dataset=dataset,
                environment=environment,
                observation=observation,
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
                    sampling_num_steps=sampling_num_steps,
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
        train_chunks = sum(len(episode) for episode in train)
        optimizer_step = math.ceil(train_chunks / config.batch_size) * config.num_epochs
        checkpoint = run_dir / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model,
            normalizer,
            config,
            optimizer_step=optimizer_step,
            dataset=dataset,
            environment=environment,
            observation=observation,
        )
        logger.log_checkpoint(checkpoint, step=optimizer_step)
    console.info(f"Saved BC run to {run_dir}")


if __name__ == "__main__":
    main()
