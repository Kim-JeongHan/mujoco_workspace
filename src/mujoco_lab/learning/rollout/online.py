"""Single-environment on-policy collection and deterministic policy evaluation."""

from collections import deque
from typing import Any, Protocol

import numpy as np
import torch
from gymnasium import spaces

from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBuffer


class RolloutEnv(Protocol):
    """Physical-action environment used by the single-environment RL trainer."""

    action_space: spaces.Box

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


def history_state(
    history: deque[np.ndarray], normalizer: Normalizer, device: torch.device
) -> torch.Tensor:
    """Normalize frames before flattening, as in the BC sequence dataset."""
    state = normalizer.normalize_state(np.stack(history)).reshape(1, -1)
    return torch.as_tensor(state, dtype=torch.float32, device=device)


def physical_action(action: torch.Tensor, normalizer: Normalizer, space: spaces.Box) -> np.ndarray:
    """Convert a single normalized chunk (1, 1, A) to bounded physical targets."""
    physical = normalizer.denormalize_action(action[0, 0].detach().cpu().numpy())
    return np.clip(physical, space.low, space.high).astype(space.dtype, copy=False)


@torch.no_grad()
def collect_rollout(
    env: RolloutEnv,
    policy: GaussianPolicy,
    critic: ValueCritic,
    normalizer: Normalizer,
    buffer: RolloutBuffer,
    *,
    steps: int,
    obs_horizon: int,
    seed: int,
) -> dict[str, float]:
    """Collect exactly ``steps`` actions, resetting episodes and the rollout boundary.

    This collector requires a single-action policy and one environment. Each
    call starts from a seeded reset, so iteration-boundary checkpoints need no
    simulator snapshot. The last unfinished transition is marked truncated.
    Next values use the post-action history before reset, and stored actions are
    the original normalized samples, independent of physical-action clipping.
    """
    if policy.chunk_size != 1 or buffer.rewards.shape[1] != 1:
        raise ValueError("online collection requires a single-action policy and one environment")
    if not 0 < steps <= buffer.rollout_length or obs_horizon <= 0:
        raise ValueError("steps must fit the buffer and obs_horizon must be positive")
    device = next(policy.parameters()).device
    buffer.reset()
    obs, _ = env.reset(seed=seed)
    history = deque([obs.copy()] * obs_horizon, maxlen=obs_horizon)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    successes = 0
    episode_return, episode_length = 0.0, 0
    completed = 0
    for step in range(steps):
        state = history_state(history, normalizer, device)
        actions = policy(state).sample()
        log_probs = policy.log_prob(state, actions)
        values = critic(state)
        obs, reward, terminated, truncated, info = env.step(
            physical_action(actions, normalizer, env.action_space)
        )
        history.append(obs.copy())
        next_values = (
            torch.zeros_like(values)
            if terminated
            else critic(history_state(history, normalizer, device))
        )
        cutoff = step + 1 == steps and not (terminated or truncated)
        buffer.add(
            states=state,
            actions=actions,
            rewards=torch.tensor([reward], dtype=values.dtype, device=device),
            values=values,
            next_values=next_values,
            log_probs=log_probs,
            terminated=torch.tensor([terminated], device=device),
            truncated=torch.tensor([truncated or cutoff], device=device),
        )
        episode_return += reward
        episode_length += 1
        if terminated or truncated:
            completed += 1
            successes += int(bool(info.get("success", False)))
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            if step + 1 < steps:
                obs, _ = env.reset(seed=seed + step + 1)
                history = deque([obs.copy()] * obs_horizon, maxlen=obs_horizon)
                episode_return, episode_length = 0.0, 0
    return {
        "rollout/steps": float(steps),
        "rollout/completed_episodes": float(completed),
        "rollout/success_rate": successes / completed if completed else 0.0,
        "rollout/mean_return": float(np.mean(episode_returns)) if completed else 0.0,
        "rollout/mean_length": float(np.mean(episode_lengths)) if completed else 0.0,
        "rollout/cutoff": float(cutoff),
    }


@torch.no_grad()
def evaluate_policy(
    env: RolloutEnv,
    policy: GaussianPolicy,
    normalizer: Normalizer,
    *,
    obs_horizon: int,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate the Gaussian mean on fixed seeds without consuming policy sampling RNG."""
    device = next(policy.parameters()).device
    successes, total_return, total_steps = 0, 0.0, 0
    for index in range(episodes):
        obs, _ = env.reset(seed=seed + index)
        history = deque([obs.copy()] * obs_horizon, maxlen=obs_horizon)
        for _ in range(max_steps):
            action = policy(history_state(history, normalizer, device)).mean
            obs, reward, terminated, truncated, info = env.step(
                physical_action(action, normalizer, env.action_space)
            )
            history.append(obs.copy())
            total_return += reward
            total_steps += 1
            if terminated or truncated:
                successes += int(bool(info.get("success", False)))
                break
    return {
        "eval/success_rate": successes / episodes,
        "eval/mean_return": total_return / episodes,
        "eval/mean_length": total_steps / episodes,
    }
