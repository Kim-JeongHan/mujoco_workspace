"""Flat configuration for offline behavior cloning experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class TrainConfig:
    """Offline BC defaults; dimensions come from the demonstration dataset."""

    # Data and scene.
    data_dir: Path = Path("data/forte_heuristic_final_100hz")
    output_dir: Path = Path("log")
    robot: Literal["panda", "forte"] = "forte"
    environment: Literal["table_shelf"] = "table_shelf"
    cubes: int = 2

    # Policy and observation/action timing.
    policy_type: Literal["mse", "flow"] = "flow"
    obs_horizon: int = 2  # Observation frames per policy input.
    chunk_size: int = 16  # Actions predicted per policy output.
    execution_horizon: int = 4  # Actions executed before the next prediction.
    physics_steps_per_action: int = 5  # Physics steps per executed action.
    flow_num_steps: int = 20  # Euler inference steps for the flow policy.
    flow_time_embed_dim: int | None = 128  # Time feature width; None uses scalar time.

    # Optimization.
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    num_epochs: int = 400
    num_workers: int = 0

    # Evaluation and logging.
    eval_interval: int = 30_000  # Optimizer steps between rollouts; 0 disables them.
    num_eval_episodes: int = 1
    eval_seed: int = 10_000
    eval_policy_seed: int = 42
    eval_max_steps: int = 1_500  # Maximum actions per evaluation episode.
    eval_xy_range: float = 0.02  # Cube offset range, in meters.
    eval_min_gap: float = 0.01  # Minimum cube edge gap, in meters.
    eval_cube_yaw_range_degrees: float | None = None  # None uses the dataset range.
    eval_video_episodes: int = 1
    eval_video_fps: int = 20
    eval_video_width: int = 640
    eval_video_height: int = 480
    # Optimizer steps between local and W&B metric logs.
    log_interval: int = 100

    # Episode-level dataset splits.
    validation_ratio: float = 0.1
    test_ratio: float = 0.0

    # Reproducibility.
    seed: int = 42
    exp_name: str | None = "forte-flow-seed42-time-embedding"

    def validate(self) -> None:
        """Check training and enabled evaluation settings before loading data."""
        if self.execution_horizon <= 0 or self.execution_horizon > self.chunk_size:
            raise ValueError("execution_horizon must be between 1 and chunk_size")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be nonnegative")
        if self.eval_interval:
            if self.num_eval_episodes <= 0:
                raise ValueError("num_eval_episodes must be positive when evaluation is enabled")
            if self.eval_max_steps <= 0:
                raise ValueError("eval_max_steps must be positive when evaluation is enabled")
            if not 0 <= self.eval_video_episodes <= self.num_eval_episodes:
                raise ValueError("eval_video_episodes must be between zero and num_eval_episodes")


@dataclass
class EvalConfig:
    """Evaluate one saved BC policy in the physical cube stacking environment."""

    checkpoint: Path
    num_episodes: int = 10
    seed: int = 5_000
    policy_seed: int = 42
    max_steps: int = 1_500  # Maximum actions per episode.
    xy_range: float = 0.02  # Cube offset range, in meters.
    min_gap: float = 0.01  # Minimum cube edge gap, in meters.
    cube_yaw_range_degrees: float | None = None  # None uses the checkpoint dataset range.
    device: Literal["cpu", "cuda"] = "cuda"
    num_video_episodes: int = 10
    video_fps: int = 20
    video_width: int = 640
    video_height: int = 480
    output_dir: Path = Path("log")
    wandb_name: str | None = None

    def validate(self) -> None:
        """Check episode and recording counts before loading a checkpoint."""
        if not 0 <= self.num_video_episodes <= self.num_episodes:
            raise ValueError("num_video_episodes must be between zero and num_episodes")
