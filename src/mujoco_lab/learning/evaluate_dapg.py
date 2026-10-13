"""Evaluate the deterministic mean of a saved DAPG policy."""

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import torch
import tyro

from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.evaluation import PolicyEvaluator
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.rl_checkpoint import load_training_checkpoint
from mujoco_lab.learning.trainers.train_dapg import (
    environment_settings,
    evaluate_training_policy,
    evaluation_config,
)


def run(
    checkpoint: Path,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    episodes: int = 10,
    seed: int = 50_000,
    max_seconds: float | None = None,
    output_dir: Path = Path("log"),
    video_episodes: int = 3,
    wandb_mode: Literal["disabled", "online"] = "disabled",
) -> dict[str, int | float]:
    """Save shared evaluation metrics and videos, with optional W&B mirroring."""
    if episodes <= 0 or video_episodes < 0:
        raise ValueError("episodes must be positive and video_episodes nonnegative")
    started = perf_counter()
    state = load_training_checkpoint(checkpoint, device=device)
    if max_seconds is not None:
        state.agent.config.max_seconds = max_seconds
    max_steps, settings = environment_settings(state)
    env, _ = create_evaluation_env(state.metadata, **settings)
    rollout = replace(
        evaluation_config(state), num_episodes=episodes, seed=seed, video_episodes=video_episodes
    )
    run_dir = output_dir / "eval" / "dapg" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    try:
        with Logger(
            run_dir,
            {
                "checkpoint": str(checkpoint.resolve()),
                "device": device,
                "rollout": asdict(rollout),
                "checkpoint_metadata": state.metadata,
            },
            wandb_mode=wandb_mode,
            run_name=f"dapg-eval-seed{seed}",
            step_metric="train/env_steps",
        ) as logger:
            rows, metrics = evaluate_training_policy(
                state, PolicyEvaluator(env, rollout, torch.device(device)), run_dir
            )
            metrics["train/env_steps"] = state.env_steps
            metrics["train/iteration"] = state.agent.iteration
            metrics["elapsed_seconds"] = perf_counter() - started
            logger.log_evaluation(metrics, rows, step=state.agent.iteration)
            return metrics
    finally:
        env.close()


def main() -> None:
    """Evaluate a resumable DAPG checkpoint through the CLI."""
    print(json.dumps(tyro.cli(run), indent=2))


if __name__ == "__main__":
    main()
