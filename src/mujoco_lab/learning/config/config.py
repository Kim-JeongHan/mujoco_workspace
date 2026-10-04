"""Configuration for behavior cloning training and evaluation."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mujoco_lab.assets import RobotName


@dataclass
class RolloutConfig:
    """Settings shared by periodic and standalone policy evaluation."""

    num_episodes: int = 3
    seed: int = 5_000
    max_steps: int = 1_500
    xy_range: float | None = None
    min_gap: float = 0.01
    cube_yaw_range_degrees: float = 45.0
    book_yaw_range_degrees: float | None = None
    video_episodes: int = 3
    video_fps: int = 20
    video_width: int = 640
    video_height: int = 480

    def validate(self) -> None:
        """Check counts required for an evaluation rollout."""
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0 <= self.video_episodes <= self.num_episodes:
            raise ValueError("video_episodes must be between zero and num_episodes")


@dataclass
class TrainConfig:
    """Offline BC defaults; dimensions come from the demonstration dataset."""

    # Data and scene.
    data_dir: Path = Path("data/forte_hurestic_cube1")
    output_dir: Path = Path("log")
    robot: RobotName = "forte"
    environment: Literal["table_shelf"] = "table_shelf"

    # Policy and observation/action timing.
    policy_type: Literal["mse", "flow"] = "flow"
    obs_horizon: int = 2  # Observation frames per policy input.
    chunk_size: int = 16  # Actions predicted per policy output.
    execution_horizon: int = 4  # Actions before replanning; 100 Hz / 4 = 25 Hz inference.
    simulation_hz: float = 500.0  # Physics and PD evaluations per simulated second.
    action_execution_hz: float = 100.0  # 500 Hz / 100 Hz = 5 physics steps/action.
    flow_num_steps: int = 20  # Euler inference steps for the flow policy.
    flow_time_embed_dim: int | None = 256  # Time feature width; None uses scalar time.

    # Optimization.
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-6
    ema_decay: float | None = 0.999  # None disables parameter averaging.
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    num_epochs: int = 400
    num_workers: int = 0

    # Evaluation and logging.
    eval_interval: int = 30_000  # Optimizer steps between evaluations; 0 disables them.
    rollout: RolloutConfig = field(default_factory=lambda: RolloutConfig())
    # Optimizer steps between local and W&B metric logs.
    log_interval: int = 100

    # Episode-level dataset splits.
    validation_ratio: float = 0.1
    test_ratio: float = 0.0

    # Reproducibility.
    seed: int = 42
    exp_name: str | None = "forte-mse-cube1"

    @property
    def simulation_dt(self) -> float:
        return 1.0 / self.simulation_hz

    @property
    def physics_steps_per_action(self) -> int:
        return int(self.simulation_hz / self.action_execution_hz)

    def validate(self) -> None:
        """Check training and enabled evaluation settings before loading data."""
        for name in ("obs_horizon", "chunk_size", "num_epochs", "log_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("simulation_hz", "action_execution_hz"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        action_repeat = self.simulation_hz / self.action_execution_hz
        if action_repeat < 1 or not math.isclose(action_repeat, round(action_repeat)):
            raise ValueError("simulation_hz must be an integer multiple of action_execution_hz")
        if (
            not math.isfinite(self.validation_ratio)
            or not math.isfinite(self.test_ratio)
            or self.validation_ratio < 0
            or self.test_ratio < 0
            or self.validation_ratio + self.test_ratio >= 1
        ):
            raise ValueError("Holdout ratios must be finite, nonnegative, and sum to less than 1")
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be None or in [0, 1)")
        if self.execution_horizon <= 0 or self.execution_horizon > self.chunk_size:
            raise ValueError("execution_horizon must be between 1 and chunk_size")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be nonnegative")
        if self.eval_interval:
            self.rollout.validate()


@dataclass
class EvalConfig:
    """Evaluate one saved BC policy in the recorded physical task environment."""

    checkpoint: Path
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    device: Literal["cpu", "cuda"] = "cuda"
    output_dir: Path = Path("log")
    wandb_name: str | None = None

    def validate(self) -> None:
        """Check episode and recording counts before loading a checkpoint."""
        self.rollout.validate()
