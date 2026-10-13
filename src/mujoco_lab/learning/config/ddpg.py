"""Single-environment DDPG training with optional future-goal replay."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mujoco_lab.assets import BookType, RobotName
from mujoco_lab.learning.config.evaluation import EvaluationConfig


@dataclass
class HERConfig:
    """Relabel this fraction of minibatch transitions with a future achieved goal."""

    ratio: float = 0.8  # Equivalent to four hindsight goals per original transition.


@dataclass
class DDPGConfig:
    """Train from scratch or resume at an episode boundary."""

    resume: Path | None = None
    output_dir: Path = Path("log")
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 42
    scene: Literal["cube_stack", "book_insertion"] = "cube_stack"
    environment: Literal["table_shelf", "warehouse"] = "table_shelf"
    robot: RobotName = "forte"
    cubes: int = 1
    book: BookType = "small"
    dt: float = 0.002
    physics_steps_per_action: int = 10
    xy_range: float = 0.02
    min_gap: float = 0.01
    cube_yaw_range_degrees: float = 0.0
    book_yaw_range_degrees: float = 0.0
    max_seconds: float = 15.0
    total_steps: int = 100_000
    obs_horizon: int = 2
    hidden_dims: tuple[int, ...] = (128, 128)
    critic_hidden_dims: tuple[int, ...] = (128, 128)
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    gamma: float = 0.98
    tau: float = 0.005  # Target = tau * online + (1 - tau) * target, per gradient step.
    batch_size: int = 256
    replay_capacity: int = 1_000_000  # Transitions; eviction retains complete episodes.
    warmup_steps: int = 2_000
    updates_per_episode: int = 40
    noise_std: float = 0.1  # Standard deviation in bounded [-1, 1] action coordinates.
    random_action_probability: float = 0.2
    her: HERConfig = field(default_factory=HERConfig)
    eval_interval_episodes: int = 10
    checkpoint_interval_episodes: int = 10
    evaluation: EvaluationConfig = field(
        default_factory=lambda: EvaluationConfig(seed=50_000, video_episodes=0)
    )
    wandb_mode: Literal["disabled", "online"] = "disabled"

    def validate(self) -> None:
        """Validate user settings once before constructing training objects."""
        if (
            min(
                self.total_steps,
                self.obs_horizon,
                self.batch_size,
                self.replay_capacity,
                self.physics_steps_per_action,
                self.updates_per_episode,
            )
            <= 0
        ):
            raise ValueError("training sizes and action cadence must be positive")
        if min(self.dt, self.max_seconds, self.actor_learning_rate, self.critic_learning_rate) <= 0:
            raise ValueError("durations and learning rates must be positive")
        if not (0 <= self.gamma < 1 and 0 < self.tau <= 1):
            raise ValueError("gamma must be in [0, 1) and tau in (0, 1]")
        if not (0 <= self.her.ratio <= 1 and 0 <= self.random_action_probability <= 1):
            raise ValueError("replay and exploration probabilities must be in [0, 1]")
        if (
            min(
                self.warmup_steps,
                self.noise_std,
                self.eval_interval_episodes,
                self.checkpoint_interval_episodes,
                self.xy_range,
                self.min_gap,
                self.cube_yaw_range_degrees,
                self.book_yaw_range_degrees,
            )
            < 0
        ):
            raise ValueError("exploration settings and intervals must be nonnegative")
        if any(width <= 0 for width in (*self.hidden_dims, *self.critic_hidden_dims)):
            raise ValueError("hidden layer widths must be positive")
        if self.evaluation.num_episodes <= 0 or self.evaluation.max_seconds <= 0:
            raise ValueError("evaluation episode count and duration must be positive")
        if self.evaluation.video_episodes < 0:
            raise ValueError("evaluation video count must be nonnegative")
