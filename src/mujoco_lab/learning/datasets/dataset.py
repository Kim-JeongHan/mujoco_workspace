from collections.abc import Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset


class ChunkDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, episodes: Sequence[Mapping[str, np.ndarray]], chunk_size: int):
        self.episodes = episodes
        self.chunk_size = chunk_size

        self.indices = []

        for ep_idx, episode in enumerate(episodes):
            T = len(episode["actions"])

            for t in range(T - chunk_size + 1):
                self.indices.append((ep_idx, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        ep_idx, t = self.indices[index]

        episode = self.episodes[ep_idx]

        state = episode["states"][t]

        action_chunk = episode["actions"][t : t + self.chunk_size]

        return state, action_chunk
