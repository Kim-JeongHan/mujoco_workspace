"""State-value network architecture and optimization settings."""

from dataclasses import dataclass


@dataclass
class CriticConfig:
    """Independent value MLP fitted to rollout return targets."""

    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate: float = 3e-4
    num_epochs: int = 5
    batch_size: int = 256
