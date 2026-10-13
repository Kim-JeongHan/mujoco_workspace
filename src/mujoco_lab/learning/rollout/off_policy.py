"""Collect owned episodes with bounded physical actions for off-policy replay."""

from collections import deque

import numpy as np
import torch

from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.envs.goals import GoalEnv
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.rollout.online import history_state
from mujoco_lab.learning.rollout.replay_buffer import ReplayEpisode


@torch.no_grad()
def collect_episode(
    env: GoalEnv,
    policy: MSEPolicy,
    normalizer: Normalizer,
    *,
    obs_horizon: int,
    max_steps: int,
    seed: int,
    env_steps: int,
    warmup_steps: int,
    noise_std: float,
    random_action_probability: float,
) -> tuple[ReplayEpisode, dict[str, float]]:
    """Collect to a physical episode boundary or an explicit action-budget cutoff.

    Observations remain raw, and actions are stored after noise and clipping in
    [-1, 1] coordinates. The final observation is copied before any reset. Original
    physical success and sparse goal-pose success are reported independently.
    """
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy()]
    history = deque((obs.copy() for _ in range(obs_horizon)), maxlen=obs_horizon)
    device = next(policy.parameters()).device
    actions, terminations, truncations = [], [], []
    success, goal_success, episode_return = False, False, 0.0
    for step in range(max_steps):
        if env_steps + step < warmup_steps or np.random.random() < random_action_probability:
            action = np.random.uniform(-1, 1, size=policy.action_dim).astype(np.float32)
        else:
            action = policy(history_state(history, normalizer, device))[0, 0].cpu().numpy()
            action = np.clip(action + np.random.normal(0, noise_std, size=action.shape), -1, 1)
            action = action.astype(np.float32)
        physical = np.clip(
            normalizer.denormalize_action(action), env.action_space.low, env.action_space.high
        )
        # Store the coordinates of the command actually sent to the environment.
        actions.append(normalizer.normalize_action(physical))
        obs, _, terminated, truncated, info = env.step(physical)
        observations.append(obs.copy())
        history.append(obs.copy())
        terminations.append(terminated)
        truncations.append(truncated or (step + 1 == max_steps and not terminated))
        reward = float(env.compute_goal_reward(env.achieved_goal(obs), env.desired_goal(obs)))
        episode_return += reward
        success |= bool(info.get("success", False))
        goal_success |= reward == 0
        if terminated or truncated:
            break
    episode = ReplayEpisode(
        np.stack(observations),
        np.stack(actions),
        np.array(terminations, dtype=bool),
        np.array(truncations, dtype=bool),
    )
    return episode, {
        "rollout/steps": float(len(actions)),
        "rollout/return": episode_return,
        "rollout/success": float(success),
        "rollout/goal_success": float(goal_success),
    }
