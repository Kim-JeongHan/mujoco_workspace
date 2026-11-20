from collections.abc import Sequence

import numpy as np
from torch.utils.data import Dataset

from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer


class ChunkDataset(Dataset[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """Pair an action chunk with oldest-to-current observation history.

    Missing early frames repeat the episode's first observation. Normalize each
    raw frame before flattening the history into one policy input vector.
    Every action-bearing observation is retained. Incomplete tail chunks use
    zero padding in normalized action coordinates and a per-action validity mask.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        chunk_size: int,
        normalizer: Normalizer | None = None,
        *,
        obs_horizon: int = 2,
    ) -> None:
        if chunk_size <= 0 or obs_horizon <= 0:
            raise ValueError("chunk_size and obs_horizon must be positive")
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.obs_horizon = obs_horizon

        self.indices: list[tuple[int, int]] = []

        for episode_idx, episode in enumerate(episodes):
            episode_len = len(episode)
            for start in range(episode_len):
                self.indices.append((episode_idx, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        episode_idx, start = self.indices[index]
        episode = self.episodes[episode_idx]

        history_indices = np.maximum(np.arange(start - self.obs_horizon + 1, start + 1), 0)
        history = episode.states[history_indices]
        actions = episode.actions[start : start + self.chunk_size]

        if self.normalizer is not None:
            history = self.normalizer.normalize_state(history)
            actions = self.normalizer.normalize_action(actions)

        valid_count = len(actions)
        action_chunk = np.zeros((self.chunk_size, episode.actions.shape[1]), dtype=np.float32)
        action_chunk[:valid_count] = actions
        valid_mask = np.arange(self.chunk_size) < valid_count
        return history.reshape(-1).astype(np.float32, copy=False), action_chunk, valid_mask
