"""Episode replay with future-goal relabeling and observation-history reconstruction."""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.envs.goals import GoalRelabeling


@dataclass
class ReplayEpisode:
    """Owned raw frames (T+1, F) and executed normalized actions (T, A)."""

    observations: np.ndarray
    actions: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


@dataclass
class ReplayBatch:
    """Normalized transition minibatch for an off-policy agent."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    relabeled: torch.Tensor


@dataclass
class ReplaySample:
    """Raw, already relabeled histories and rewards before normalization."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    relabeled: np.ndarray

    def to_batch(self, normalizer: Normalizer, device: str | torch.device) -> ReplayBatch:
        """Normalize goal-consistent histories and transfer the sampled minibatch."""
        size = len(self.actions)
        return ReplayBatch(
            torch.as_tensor(
                normalizer.normalize_state(self.states).reshape(size, -1),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(self.actions, dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            torch.as_tensor(
                normalizer.normalize_state(self.next_states).reshape(size, -1),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(self.terminated, device=device),
            torch.as_tensor(self.truncated, device=device),
            torch.as_tensor(self.relabeled, device=device),
        )


class HERReplayBuffer:
    """Sample transitions uniformly and relabel goals without duplicating episodes."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.episodes: deque[ReplayEpisode] = deque()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, episode: ReplayEpisode) -> None:
        """Take ownership of a complete episode, evicting complete oldest episodes."""
        size = len(episode.actions)
        if size == 0 or size > self.capacity:
            raise ValueError("episode length must be positive and fit the replay capacity")
        if len(episode.observations) != size + 1:
            raise ValueError("replay requires the final observation before reset")
        while self._size + size > self.capacity:
            self._size -= len(self.episodes.popleft().actions)
        self.episodes.append(episode)
        self._size += size

    def sample(
        self, batch_size: int, env: GoalRelabeling, *, obs_horizon: int, her_ratio: float
    ) -> ReplaySample:
        """Relabel entire histories with one future goal from the same episode.

        Future indices include the transition's next state and the episode's final
        state. Pose success is nonterminal; stored physical termination flags are
        goal-independent, and truncation preserves next-state bootstrapping.
        """
        if not self.episodes:
            raise RuntimeError("collect an episode before sampling replay")
        episodes = tuple(self.episodes)
        ends = np.cumsum([len(episode.actions) for episode in episodes])
        indices = np.random.randint(self._size, size=batch_size)
        episode_indices = np.searchsorted(ends, indices, side="right")
        starts = np.r_[0, ends[:-1]]
        steps = indices - starts[episode_indices]
        relabeled = np.random.random(batch_size) < her_ratio
        histories, next_histories, actions, terminated, truncated = [], [], [], [], []
        offsets = np.arange(1 - obs_horizon, 1)
        for index, episode_index in enumerate(episode_indices):
            episode = episodes[episode_index]
            step = steps[index]
            goal = env.desired_goal(episode.observations[step])
            if relabeled[index]:
                future = np.random.randint(step + 1, len(episode.observations))
                goal = env.achieved_goal(episode.observations[future])
            histories.append(
                env.relabel_observation(episode.observations[np.maximum(step + offsets, 0)], goal)
            )
            next_histories.append(
                env.relabel_observation(
                    episode.observations[np.maximum(step + 1 + offsets, 0)], goal
                )
            )
            actions.append(episode.actions[step])
            terminated.append(episode.terminated[step])
            truncated.append(episode.truncated[step])
        states = np.stack(histories)
        next_states = np.stack(next_histories)
        rewards = env.compute_goal_reward(
            env.achieved_goal(next_states[:, -1]), env.desired_goal(next_states[:, -1])
        )
        return ReplaySample(
            states,
            np.stack(actions),
            rewards,
            next_states,
            np.array(terminated, dtype=bool),
            np.array(truncated, dtype=bool),
            relabeled,
        )

    def state_dict(self) -> dict[str, Any]:
        """Save owned episodes as tensors accepted by weights-only loading."""
        return {
            "capacity": self.capacity,
            "episodes": [
                {
                    name: torch.from_numpy(getattr(episode, name).copy())
                    for name in ("observations", "actions", "terminated", "truncated")
                }
                for episode in self.episodes
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay contents in their original eviction order."""
        self.capacity = state["capacity"]
        self.episodes.clear()
        self._size = 0
        for episode in state["episodes"]:
            self.add(ReplayEpisode(**{name: tensor.numpy() for name, tensor in episode.items()}))
