"""On-policy rollout storage and shuffled minibatches for policy updates."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """Flattened transitions with fixed values and log probabilities from collection."""

    states: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Store one rollout, reuse it for update epochs, then reset for a new policy.

    Storage has leading axes (T, E): rollout steps and environments. Each call
    to ``add`` writes one step for all environments. States have shape
    (E, state_dim), actions (E, *action_shape), and scalar fields (E,).
    ``action_shape`` can describe a single action or an entire action chunk.

    Callers supply compatible shapes, dtypes, and devices. Store the original
    sampled actions before environment clipping, and their joint log probability
    reduced over action dimensions. ``next_values`` uses the final observation
    before reset at episode boundaries. Inputs are copied without autograd.
    Compute advantages and returns outside the buffer using the stored prefix
    ``[:len(buffer)]``, then pass them to ``set_targets`` before sampling batches.
    """

    def __init__(
        self,
        rollout_length: int,
        state_dim: int,
        action_shape: tuple[int, ...],
        *,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.rollout_length = rollout_length
        self.pos = 0
        self._targets_ready = False
        shape = (rollout_length, num_envs)
        self.states = torch.empty((*shape, state_dim), device=device, dtype=dtype)
        self.actions = torch.empty((*shape, *action_shape), device=device, dtype=dtype)
        self.rewards = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.next_values = torch.empty(shape, device=device, dtype=dtype)
        self.log_probs = torch.empty(shape, device=device, dtype=dtype)
        self.terminated = torch.empty(shape, device=device, dtype=torch.bool)
        self.truncated = torch.empty(shape, device=device, dtype=torch.bool)
        self.advantages = torch.empty(shape, device=device, dtype=dtype)
        self.returns = torch.empty(shape, device=device, dtype=dtype)

    def __len__(self) -> int:
        """Return the number of stored time steps, each containing E transitions."""
        return self.pos

    def reset(self) -> None:
        """Discard the rollout and reuse its allocated storage."""
        self.pos = 0
        self._targets_ready = False

    @torch.no_grad()
    def add(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        log_probs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Copy one step's transitions; adding invalidates previously computed targets."""
        if self.pos >= self.rollout_length:
            raise RuntimeError("rollout buffer is full; reset before collecting a new rollout")
        self.states[self.pos].copy_(states)
        self.actions[self.pos].copy_(actions)
        self.rewards[self.pos].copy_(rewards)
        self.values[self.pos].copy_(values)
        self.next_values[self.pos].copy_(next_values)
        self.log_probs[self.pos].copy_(log_probs)
        self.terminated[self.pos].copy_(terminated)
        self.truncated[self.pos].copy_(truncated)
        self.pos += 1
        self._targets_ready = False

    @torch.no_grad()
    def set_targets(self, advantages: torch.Tensor, returns: torch.Tensor) -> None:
        """Copy externally computed targets of shape (len(buffer), E) without gradients.

        The caller chooses the advantage estimator and critic target separately.
        Any advantage normalization also belongs to the caller.
        """
        if self.pos == 0:
            raise RuntimeError("cannot set targets for an empty rollout")
        shape = self.rewards[: self.pos].shape
        if advantages.shape != shape or returns.shape != shape:
            raise ValueError("targets must have shape (len(buffer), num_envs)")
        self.advantages[: self.pos].copy_(advantages)
        self.returns[: self.pos].copy_(returns)
        self._targets_ready = True

    def minibatches(self, batch_size: int) -> Iterator[RolloutBatch]:
        """Shuffle all stored transitions once, retaining the final smaller batch.

        Call again for each update epoch to reshuffle the same rollout. Do not
        modify or reset the buffer while iterating. Targets are returned as supplied.
        """
        if not self._targets_ready:
            raise RuntimeError("set targets before requesting minibatches")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        states = self.states[: self.pos].flatten(0, 1)
        actions = self.actions[: self.pos].flatten(0, 1)
        values = self.values[: self.pos].flatten(0, 1)
        log_probs = self.log_probs[: self.pos].flatten(0, 1)
        advantages = self.advantages[: self.pos].flatten(0, 1)
        returns = self.returns[: self.pos].flatten(0, 1)
        indices = torch.randperm(states.shape[0], device=states.device)
        for start in range(0, indices.numel(), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield RolloutBatch(
                states=states[batch_indices],
                actions=actions[batch_indices],
                old_values=values[batch_indices],
                old_log_probs=log_probs[batch_indices],
                advantages=advantages[batch_indices],
                returns=returns[batch_indices],
            )
