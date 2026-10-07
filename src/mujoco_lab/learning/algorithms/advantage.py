"""Advantage and return estimation shared by policy gradient algorithms."""

import torch


@torch.no_grad()
def compute_discounted_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float = 0.99,
    next_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum discounted rewards within each episode in a rollout.

    Inputs have the same nonempty shape (T,) or (T, E). Rewards use a floating
    dtype, and boundary flags are boolean tensors on the same device.
    Callers provide valid inputs and gamma in [0, 1].

    By default, returns use only observed rewards, with zero continuation at
    episode boundaries and the rollout cutoff. This matches the reward-to-go
    targets used by the reference DAPG implementation.

    Optionally supply ``next_values`` with the rewards' shape, dtype, and device
    to bootstrap at truncations and the rollout cutoff. These values must refer
    to the next observation before any reset. True termination always disables
    bootstrapping. Returns never include rewards from a subsequent episode.
    Outputs are detached from autograd, and inputs are left unchanged.
    """
    bootstrap_values = (
        torch.zeros_like(rewards)
        if next_values is None
        else torch.where(terminated, 0.0, next_values)
    )
    episode_ends = terminated | truncated
    returns = torch.empty_like(rewards)
    next_return = bootstrap_values[-1]
    for step in range(rewards.shape[0] - 1, -1, -1):
        continuation = torch.where(episode_ends[step], bootstrap_values[step], next_return)
        next_return = rewards[step] + gamma * continuation
        returns[step] = next_return
    return returns


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
) -> torch.Tensor:
    """Compute unnormalized GAE advantages independently of critic targets.

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
    with the masks described above. Outputs are detached from autograd, and the
    inputs are left unchanged. Compute critic targets separately with
    ``compute_discounted_returns``, or add detached values for lambda-returns.
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
    return advantages
