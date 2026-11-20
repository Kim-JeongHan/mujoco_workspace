"""Train DDPG with future-goal HER in the existing physical manipulation environments."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
import tyro

from mujoco_lab.learning.agents.ddpg import DDPGAgent
from mujoco_lab.learning.checkpoint import (
    DDPGTrainingState,
    capture_rng,
    config_dict,
    load_ddpg_checkpoint,
    restore_rng,
    save_ddpg_checkpoint,
)
from mujoco_lab.learning.config.ddpg import DDPGConfig
from mujoco_lab.learning.config.replay import BookReplayConfig, CubeStackReplayConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.envs.factory import create_env
from mujoco_lab.learning.envs.goals import GoalEnv
from mujoco_lab.learning.envs.joint_action_env import JointActionEnv
from mujoco_lab.learning.evaluate import create_rollout_env
from mujoco_lab.learning.evaluation import PolicyEvaluator, evaluation_log_metrics
from mujoco_lab.learning.infrastructure.utils import set_seed
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.metadata import (
    CheckpointMetadata,
    EnvironmentMetadata,
    ObservationMetadata,
    PolicyMetadata,
)
from mujoco_lab.learning.models.critic import QCritic
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.rollout.off_policy import collect_episode
from mujoco_lab.learning.timing import max_steps_for_seconds


def scene_config(config: DDPGConfig) -> CubeStackReplayConfig | BookReplayConfig:
    """Describe the fresh training environment from user configuration."""
    if config.scene == "cube_stack":
        return CubeStackReplayConfig(
            robot=config.robot,
            cubes=config.cubes,
            environment=config.environment,
            physics_steps_per_action=config.physics_steps_per_action,
            xy_range=config.xy_range,
            min_gap=config.min_gap,
            cube_yaw_range_degrees=config.cube_yaw_range_degrees,
        )
    return BookReplayConfig(
        robot=config.robot,
        book=config.book,
        physics_steps_per_action=config.physics_steps_per_action,
        xy_range=config.xy_range,
        book_yaw_range_degrees=config.book_yaw_range_degrees,
    )


def initialize_training(
    config: DDPGConfig, env: JointActionEnv, environment: EnvironmentMetadata
) -> DDPGTrainingState:
    """Initialize bounded actions, online frame statistics, and independent Q weights."""
    frame_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = MSEPolicy(
        frame_dim * config.obs_horizon, action_dim, 1, config.hidden_dims, bounded=True
    ).to(config.device)
    critic = QCritic(policy.state_dim, action_dim, config.critic_hidden_dims).to(config.device)
    normalizer = Normalizer(
        np.zeros(frame_dim, dtype=np.float32),
        np.ones(frame_dim, dtype=np.float32),
        (env.action_space.low + env.action_space.high) / 2,
        (env.action_space.high - env.action_space.low) / 2,
        state_clip=5.0,
    )
    metadata = CheckpointMetadata(
        policy=PolicyMetadata(
            policy_type="mse",
            bounded_actions=True,
            state_dim=policy.state_dim,
            action_dim=action_dim,
            obs_horizon=config.obs_horizon,
            hidden_dims=config.hidden_dims,
        ),
        environment=environment,
        observation=ObservationMetadata.from_dict(env.observation_metadata),
        algorithm="ddpg",
    )
    return DDPGTrainingState(
        DDPGAgent(policy, critic, config),
        normalizer,
        metadata,
    )


def train_episode(state: DDPGTrainingState, env: GoalEnv, *, max_steps: int) -> dict[str, float]:
    """Collect one episode, update raw/relabelled statistics, and train from replay."""
    config = state.agent.config
    episode, metrics = collect_episode(
        env,
        state.agent.policy,
        state.normalizer,
        obs_horizon=state.metadata.policy.obs_horizon,
        max_steps=min(max_steps, config.total_steps - state.env_steps),
        seed=config.seed + state.episodes,
        env_steps=state.env_steps,
        warmup_steps=config.warmup_steps,
        noise_std=config.noise_std,
        random_action_probability=config.random_action_probability,
    )
    state.replay.add(episode)
    state.moments.update(episode.observations, state.normalizer)
    state.env_steps += len(episode.actions)
    state.episodes += 1
    sample = state.replay.sample(
        config.batch_size,
        env,
        obs_horizon=state.metadata.policy.obs_horizon,
        her_ratio=config.her.ratio,
    )
    # Include virtual goals in scaling, instead of fitting only the original fixed goal.
    state.moments.update(
        np.concatenate((sample.states[:, -1], sample.next_states[:, -1])), state.normalizer
    )
    if state.env_steps >= config.warmup_steps:
        updates = []
        for index in range(config.updates_per_episode):
            if index:
                sample = state.replay.sample(
                    config.batch_size,
                    env,
                    obs_horizon=state.metadata.policy.obs_horizon,
                    her_ratio=config.her.ratio,
                )
            updates.append(state.agent.update(sample.to_batch(state.normalizer, config.device)))
        metrics.update({key: float(np.mean([row[key] for row in updates])) for key in updates[0]})
    metrics.update(
        {
            "train/env_steps": float(state.env_steps),
            "train/episodes": float(state.episodes),
            "train/iteration": float(state.agent.iteration),
            "replay/size": float(len(state.replay)),
        }
    )
    return metrics


def run(config: DDPGConfig) -> Path:
    """Run to an action budget and save resumable episode-boundary checkpoints.

    Resume retains saved learning and scene settings. Only the total action
    budget, output directory, device, and W&B mode come from the new invocation.
    Training uses fixed horizons; evaluation retains released stable placement.
    """
    config.validate()
    set_seed(config.seed)
    state = None
    if config.resume is not None:
        state = load_ddpg_checkpoint(config.resume, device=config.device)
        config = replace(
            state.agent.config,
            resume=config.resume,
            total_steps=config.total_steps,
            output_dir=config.output_dir,
            device=config.device,
            wandb_mode=config.wandb_mode,
        )
        config.validate()
        state.agent.config = config
    if state is not None and config.total_steps <= state.env_steps:
        raise ValueError("total_steps must exceed the checkpoint's collected steps")
    environment = (
        state.metadata.environment
        if state is not None
        else EnvironmentMetadata(scene_config(config), config.dt, config.robot)
    )
    max_steps = max_steps_for_seconds(config.max_seconds, environment.action_dt)
    if config.replay_capacity < max_steps:
        raise ValueError("replay_capacity must fit a full training episode")
    env = create_env(
        environment.scene,
        dt=environment.dt,
        robot_name=environment.robot_name,
        max_steps=max_steps,
        terminate_on_success=False,
    )
    eval_env = None
    evaluator = None
    run_dir = config.output_dir / "ddpg" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    try:
        if state is None:
            state = initialize_training(config, env, environment)
        if config.eval_interval_episodes:
            eval_env, _ = create_rollout_env(environment, config.evaluation)
            evaluator = PolicyEvaluator(eval_env, config.evaluation, torch.device(config.device))
        with Logger(
            run_dir,
            {**config_dict(config), "metadata": state.metadata.to_dict()},
            wandb_mode=config.wandb_mode,
            step_metric="train/env_steps",
        ) as logger:
            if state.rng is not None:
                restore_rng(state.rng)
                state.rng = None
            started = perf_counter()
            while state.env_steps < config.total_steps:
                metrics = train_episode(state, env, max_steps=max_steps)
                episodes = None
                if evaluator is not None and (
                    state.episodes % config.eval_interval_episodes == 0
                    or state.env_steps == config.total_steps
                ):
                    rng = capture_rng()
                    try:
                        episodes, summary = evaluator.evaluate_to_directory(
                            state.agent.policy,
                            state.normalizer,
                            state.metadata,
                            run_dir / "eval" / f"step_{state.env_steps:08d}",
                        )
                        metrics.update(evaluation_log_metrics(summary))
                    finally:
                        restore_rng(rng)
                metrics["elapsed_seconds"] = perf_counter() - started
                if episodes is None:
                    logger.log(metrics, step=state.agent.iteration)
                else:
                    logger.log_evaluation(metrics, episodes, step=state.agent.iteration)
                if (
                    config.checkpoint_interval_episodes
                    and state.episodes % config.checkpoint_interval_episodes == 0
                ):
                    path = run_dir / f"checkpoint_{state.episodes:06d}.pt"
                    save_ddpg_checkpoint(path, state)
                    logger.log_checkpoint(path, step=state.agent.iteration)
            path = run_dir / "checkpoint.pt"
            save_ddpg_checkpoint(path, state)
            logger.log_checkpoint(path, step=state.agent.iteration)
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
    return run_dir


def main() -> None:
    """CLI entry point for DDPG+HER training, ablation, and resume."""
    config = tyro.cli(DDPGConfig, description=__doc__)
    print(f"Saved DDPG training to {run(config)}")


if __name__ == "__main__":
    main()
