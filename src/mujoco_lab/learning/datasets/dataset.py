from torch.utils.data import Dataset

from mujoco_lab.learning.datasets.episode import Episode


class ChunkDataset(Dataset):
    def __init__(self, episodes: Episode, chunk_size: float):
        self.episodes = episodes
        self.chunk_size = chunk_size

        self.indices = []

        for ep_idx, episode in enumerate(episodes):
            T = len(episode["actions"])

            for t in range(T - chunk_size + 1):
                self.indices.append((ep_idx, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]

        episode = self.episodes[ep_idx]

        state = episode["states"][t]

        action_chunk = episode["actions"][t : t + self.chunk_size]

        return state, action_chunk
