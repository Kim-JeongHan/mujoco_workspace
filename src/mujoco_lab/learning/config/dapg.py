"""Configuration for single-action DAPG fine-tuning."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mujoco_lab.learning.config.critic import CriticConfig
from mujoco_lab.learning.config.npg import NPGConfig


@dataclass
class DAPGConfig:
    """Start from an MSE BC checkpoint or resume an iteration-boundary RL checkpoint."""

    init_from: Path | None = None
    resume: Path | None = None
    data_dir: Path = Path("data/forte_mixed_1")
    output_dir: Path = Path("log")
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 42
    total_steps: int = 100_000  # Total collected actions, including resumed steps.
    rollout_steps: int = 2048
    max_seconds: float = 15.0
    gamma: float = 0.995
    gae_lambda: float = 0.97
    bootstrap_returns: bool = True  # Retain next-state value at fixed-length rollout cutoffs.
    initial_arm_std_degrees: float = 0.5  # Initial joint-target exploration standard deviation.
    initial_gripper_std_m: float = 0.0005  # Initial opening-width exploration standard deviation.
    demo_weight: float = 0.1
    demo_decay: float = 0.95
    demo_batch_size: int = 0  # Zero uses every demonstration transition.
    npg: NPGConfig = field(default_factory=NPGConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    eval_interval_iterations: int = 10  # Training iterations; zero disables evaluation.
    eval_episodes: int = 3
    eval_seed: int = 50_000
    eval_video_episodes: int = 3  # Record the first evaluation seeds, as in BC.
    checkpoint_interval_iterations: int = 10  # The final checkpoint is always saved.
    wandb_mode: Literal["disabled", "online"] = "disabled"

    def validate(self) -> None:
        """Validate CLI settings once before loading data or constructing environments."""
        if (self.init_from is None) == (self.resume is None):
            raise ValueError("provide exactly one of init_from or resume")
        if (
            min(
                self.total_steps, self.rollout_steps, self.critic.num_epochs, self.critic.batch_size
            )
            <= 0
        ):
            raise ValueError("step budgets and critic training sizes must be positive")
        if self.max_seconds <= 0 or self.eval_episodes <= 0:
            raise ValueError("max_seconds and eval_episodes must be positive")
        if self.initial_arm_std_degrees <= 0 or self.initial_gripper_std_m <= 0:
            raise ValueError("initial exploration standard deviations must be positive")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 <= self.demo_decay <= 1):
            raise ValueError("discount and decay factors must be in [0, 1]")
        if (
            min(
                self.demo_weight,
                self.demo_batch_size,
                self.eval_interval_iterations,
                self.eval_video_episodes,
                self.checkpoint_interval_iterations,
            )
            < 0
        ):
            raise ValueError("demo settings and logging intervals must be nonnegative")
