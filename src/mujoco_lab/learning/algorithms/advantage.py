"""Advantage and return estimation shared by policy gradient algorithms."""

import torch


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unnormalized GAE advantages and lambda-return value targets.

    All inputs have shape (T,) for one environment or (T, E) for E environments,
    with a nonempty time axis. Rewards and values share a floating dtype and
    device; termination flags are boolean tensors on that device.
    ``gamma`` and ``gae_lambda`` are in [0, 1]. Callers provide valid inputs.

    ``next_values[t]`` is the value of the observation reached by transition t,
    including the final observation before an automatic reset. True termination
    disables bootstrapping; time-limit truncation retains it. Both stop the GAE
    recurrence from crossing an episode boundary. The final rollout transition
    bootstraps from its next value without requiring the episode to end.

    The recurrence is delta[t] = reward[t] + gamma * next_value[t] - value[t],
    followed by advantage[t] = delta[t] + gamma * lambda * advantage[t + 1],
    with the masks described above. Targets equal advantages + values. Outputs
    are detached from autograd, and the inputs are left unchanged.
    """
    bootstrap_values = torch.where(terminated, 0.0, next_values)
    deltas = rewards + gamma * bootstrap_values - values
    episode_ends = terminated | truncated
    advantages = torch.empty_like(values)
    next_advantage = torch.zeros_like(values[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        continuation = torch.where(episode_ends[step], 0.0, next_advantage)
        next_advantage = deltas[step] + gamma * gae_lambda * continuation
        advantages[step] = next_advantage
    return advantages, advantages + values
