"""Evaluate the deterministic mean of a saved DAPG policy."""

import json
from pathlib import Path
from typing import Literal

import tyro

from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.rl_checkpoint import load_training_checkpoint
from mujoco_lab.learning.rollout.online import evaluate_policy
from mujoco_lab.learning.trainers.train_dapg import environment_settings


def run(
    checkpoint: Path,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    episodes: int = 10,
    seed: int = 50_000,
    max_seconds: float | None = None,
) -> dict[str, float]:
    """Use the checkpoint scene and action cadence, printing scalar evaluation results."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    state = load_training_checkpoint(checkpoint, device=device)
    if max_seconds is not None:
        state.config.max_seconds = max_seconds
    max_steps, settings = environment_settings(state)
    env, _ = create_evaluation_env(state.metadata, **settings)
    try:
        return evaluate_policy(
            env,
            state.actor,
            state.normalizer,
            obs_horizon=state.metadata["architecture"]["obs_horizon"],
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
        )
    finally:
        env.close()


def main() -> None:
    """Evaluate a resumable DAPG checkpoint through the CLI."""
    print(json.dumps(tyro.cli(run), indent=2))


if __name__ == "__main__":
    main()
