"""Fine-tune a BC-initialized single-action Gaussian policy with DAPG."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import tyro

from mujoco_lab.learning.algorithms.advantage import compute_discounted_returns, compute_gae
from mujoco_lab.learning.algorithms.dapg import Demonstrations, dapg_step
from mujoco_lab.learning.config.dapg import DAPGConfig
from mujoco_lab.learning.datasets.episode import load_episodes
from mujoco_lab.learning.datasets.replay import require_width_actions
from mujoco_lab.learning.datasets.sequence import ChunkDataset
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.infrastructure.utils import set_seed
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rl_checkpoint import (
    TrainingState,
    config_dict,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
)
from mujoco_lab.learning.rollout.buffer import RolloutBuffer
from mujoco_lab.learning.rollout.online import RolloutEnv, collect_rollout, evaluate_policy
from mujoco_lab.learning.timing import max_steps_for_seconds
from mujoco_lab.learning.trainers.train_ppo import initialize_actor


def initialize_training(config: DAPGConfig) -> TrainingState:
    """Use the first BC action head and retain its observation normalization/history."""
    if config.init_from is None:
        raise ValueError("init_from is required for fresh training")
    source, normalizer, metadata = initialize_actor(
        config.init_from, initial_log_std=config.initial_log_std, device=config.device
    )
    architecture = metadata["architecture"]
    actor = GaussianPolicy(
        source.state_dim, source.action_dim, 1, tuple(architecture["hidden_dims"])
    ).to(config.device)
    actor.net.load_state_dict(source.net.state_dict())
    actor.mean_head.load_state_dict(
        {name: value[: source.action_dim] for name, value in source.mean_head.state_dict().items()}
    )
    with torch.no_grad():
        actor.log_std.copy_(source.log_std[:1])
    metadata = deepcopy(metadata)
    metadata["source_bc_architecture"] = dict(architecture)
    metadata["architecture"].update(policy_type="gaussian", chunk_size=1, execution_horizon=1)
    critic = ValueCritic(
        actor.state_dim,
        config.critic_n_layers,
        config.critic_layer_size,
        config.critic_learning_rate,
        device=config.device,
    )
    return TrainingState(actor, critic, normalizer, metadata, config)


def load_demonstrations(state: TrainingState) -> Demonstrations:
    """Load successful BC training episodes with matching scene, cadence, and observations."""
    episodes = load_episodes(state.config.data_dir)
    dataset = state.metadata["dataset_metadata"]
    architecture = state.metadata["architecture"]
    replay = dataset["replay"]
    train_seeds = dataset.get("train_seeds")
    if train_seeds is not None:
        seeds = set(train_seeds)
        episodes = [episode for episode in episodes if episode.metadata.get("seed") in seeds]
    if not episodes:
        raise ValueError(
            "no BC training demonstrations remain after filtering recorded train seeds"
        )
    for episode in episodes:
        recorded = episode.metadata["replay"]
        require_width_actions(recorded)
        episode.check_physics_step_consistency(
            architecture["physics_steps_per_action"], simulation_dt=replay["dt"]
        )
        keys = ("scene", "robot", "robot_name", "environment", "cubes", "book")
        if any(recorded.get(key) != replay.get(key) for key in keys):
            raise ValueError("demonstration scene differs from the BC checkpoint")
        if (
            episode.states.shape[1] != len(state.normalizer.state_mean)
            or episode.actions.shape[1] != state.actor.action_dim
            or episode.metadata["observation"] != dataset["observation"]
        ):
            raise ValueError(
                "demonstration observation/action layout differs from the BC checkpoint"
            )
    chunks = ChunkDataset(episodes, 1, state.normalizer, obs_horizon=architecture["obs_horizon"])
    observations, actions = zip(*(chunks[index] for index in range(len(chunks))), strict=True)
    return Demonstrations(
        torch.from_numpy(np.stack(observations)).to(state.config.device),
        torch.from_numpy(np.stack(actions)).to(state.config.device),
    )


def train_iteration(
    state: TrainingState, env: RolloutEnv, demos: Demonstrations, *, steps: int
) -> dict[str, float]:
    """Collect, estimate targets, update the actor once, then fit the critic."""
    config = state.config
    actor, critic = state.actor, state.critic
    buffer = RolloutBuffer(steps, actor.state_dim, (1, actor.action_dim), device=config.device)
    metrics = collect_rollout(
        env,
        actor,
        critic,
        state.normalizer,
        buffer,
        steps=steps,
        obs_horizon=state.metadata["architecture"]["obs_horizon"],
        seed=config.seed + state.env_steps,
    )
    advantages = compute_gae(
        buffer.rewards,
        buffer.values,
        buffer.next_values,
        buffer.terminated,
        buffer.truncated,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    returns = compute_discounted_returns(
        buffer.rewards,
        buffer.terminated,
        buffer.truncated,
        gamma=config.gamma,
        next_values=buffer.next_values if config.bootstrap_returns else None,
    )
    buffer.set_targets(advantages, returns)
    batch = next(buffer.minibatches(steps))
    metrics.update(
        dapg_step(
            actor,
            batch,
            demos,
            iteration=state.iteration,
            config=config.npg,
            demo_weight=config.demo_weight,
            demo_decay=config.demo_decay,
            demo_batch_size=config.demo_batch_size,
        )
    )
    critic_losses = []
    for _ in range(config.critic_epochs):
        for minibatch in buffer.minibatches(config.critic_batch_size):
            critic_losses.append(
                critic.update(minibatch.states, minibatch.returns)["Baseline Loss"]
            )
    metrics["critic/loss"] = float(np.mean(critic_losses))
    state.iteration += 1
    state.env_steps += steps
    metrics["train/env_steps"] = float(state.env_steps)
    return metrics


def environment_settings(state: TrainingState) -> tuple[int, dict[str, Any]]:
    """Preserve the BC scene, randomization, and physical action cadence."""
    replay = state.metadata["dataset_metadata"]["replay"]
    dt = replay["dt"] * state.metadata["architecture"]["physics_steps_per_action"]
    max_steps = max_steps_for_seconds(state.config.max_seconds, dt)
    return max_steps, {
        "xy_range": None,
        "min_gap": None,
        "max_steps": max_steps,
        "cube_yaw_range_degrees": None,
        "book_yaw_range_degrees": None,
    }


def run(config: DAPGConfig) -> Path:
    """Run to an action budget, saving evaluation metrics and resumable checkpoints.

    Resume restores saved learning settings and dataset paths. Only total_steps,
    output_dir, device, and wandb_mode are taken from the new command. Each resume
    writes a fresh run directory and starts at a seeded rollout boundary.
    """
    config.validate()
    set_seed(config.seed)
    if config.resume is None:
        state = initialize_training(config)
    else:
        state = load_training_checkpoint(config.resume, device=config.device)
        config = replace(
            state.config,
            init_from=None,
            resume=config.resume,
            total_steps=config.total_steps,
            output_dir=config.output_dir,
            device=config.device,
            wandb_mode=config.wandb_mode,
        )
        state.config = config
    if config.total_steps <= state.env_steps:
        raise ValueError("total_steps must exceed the checkpoint's collected steps")
    demos = load_demonstrations(state)
    max_steps, settings = environment_settings(state)
    env, _ = create_evaluation_env(state.metadata, **settings)
    eval_env = None
    run_dir = config.output_dir / "dapg" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    try:
        if config.eval_interval:
            eval_env, _ = create_evaluation_env(state.metadata, **settings)
        with Logger(run_dir, config_dict(config), wandb_mode=config.wandb_mode) as logger:
            if state.rng is not None:
                restore_rng(state.rng)
                state.rng = None
            while state.env_steps < config.total_steps:
                metrics = train_iteration(
                    state,
                    env,
                    demos,
                    steps=min(config.rollout_steps, config.total_steps - state.env_steps),
                )
                if eval_env is not None and (
                    state.iteration % config.eval_interval == 0
                    or state.env_steps == config.total_steps
                ):
                    metrics.update(
                        evaluate_policy(
                            eval_env,
                            state.actor,
                            state.normalizer,
                            obs_horizon=state.metadata["architecture"]["obs_horizon"],
                            episodes=config.eval_episodes,
                            max_steps=max_steps,
                            seed=config.eval_seed,
                        )
                    )
                logger.log(metrics, step=state.iteration)
                if config.checkpoint_interval and state.iteration % config.checkpoint_interval == 0:
                    path = run_dir / f"checkpoint_{state.iteration:06d}.pt"
                    save_training_checkpoint(path, state)
                    logger.log_checkpoint(path, step=state.iteration)
            path = run_dir / "checkpoint.pt"
            save_training_checkpoint(path, state)
            logger.log_checkpoint(path, step=state.iteration)
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
    return run_dir


def main() -> None:
    """CLI entry point for DAPG fine-tuning and resume."""
    config = tyro.cli(DAPGConfig, description=__doc__)
    print(f"Saved DAPG training to {run(config)}")


if __name__ == "__main__":
    main()
