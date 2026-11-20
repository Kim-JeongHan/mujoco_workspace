"""Evaluate a saved bounded DDPG policy with physical task success metrics."""

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
import tyro

from mujoco_lab.learning.checkpoint import load_ddpg_checkpoint
from mujoco_lab.learning.evaluate import create_rollout_env
from mujoco_lab.learning.evaluation import PolicyEvaluator, evaluation_log_metrics
from mujoco_lab.learning.logging import Logger


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
    """Save goal-pose success and released-placement success without exploration noise."""
    if episodes <= 0 or video_episodes < 0:
        raise ValueError("episodes must be positive and video_episodes nonnegative")
    state = load_ddpg_checkpoint(checkpoint, device=device)
    config = state.agent.config
    rollout = replace(
        config.evaluation,
        num_episodes=episodes,
        seed=seed,
        max_seconds=config.evaluation.max_seconds if max_seconds is None else max_seconds,
        video_episodes=video_episodes,
    )
    env, _ = create_rollout_env(state.metadata.environment, rollout)
    run_dir = output_dir / "eval" / "ddpg" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    try:
        with Logger(
            run_dir,
            {
                "checkpoint": str(checkpoint.resolve()),
                "device": device,
                "rollout": asdict(rollout),
                "metadata": state.metadata.to_dict(),
            },
            wandb_mode=wandb_mode,
            step_metric="train/env_steps",
        ) as logger:
            rows, summary = PolicyEvaluator(
                env, rollout, torch.device(device)
            ).evaluate_to_directory(
                state.agent.policy,
                state.normalizer,
                state.metadata,
                run_dir,
            )
            metrics = evaluation_log_metrics(summary)
            metrics["train/env_steps"] = state.env_steps
            metrics["train/iteration"] = state.agent.iteration
            logger.log_evaluation(metrics, rows, step=state.agent.iteration)
            return metrics
    finally:
        env.close()


def main() -> None:
    print(json.dumps(tyro.cli(run), indent=2))


if __name__ == "__main__":
    main()
