"""Configuration for behavior cloning training and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mujoco_lab.assets import RobotName
from mujoco_lab.learning.timing import physics_steps_per_action


@dataclass
class RolloutConfig:
    """Settings shared by periodic and standalone policy evaluation."""

    num_episodes: int = 3
    seed: int = 5_000
    max_seconds: float = 15.0  # Maximum simulated duration of each evaluation episode.
    xy_range: float | None = 0.02
    min_gap: float = 0.01
    cube_yaw_range_degrees: float = 45.0
    book_yaw_range_degrees: float | None = None
    video_episodes: int = 3
    video_fps: int = 20
    video_width: int = 640
    video_height: int = 480


@dataclass
class TrainConfig:
    """Offline BC defaults; dimensions come from the demonstration dataset."""

    # Data and scene.
    data_dir: Path = Path("data/forte_mixed_1")
    output_dir: Path = Path("log")
    robot: RobotName = "forte"

    # Policy and observation/action timing.
    policy_type: Literal["mse", "flow"] = "mse"
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
    eval_interval: int = 30_000  # Optimizer steps between evaluations; nonpositive disables them.
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
        return physics_steps_per_action(self.simulation_hz, self.action_execution_hz)


@dataclass
class EvalConfig:
    """Evaluate one saved BC policy in the recorded physical task environment."""

    checkpoint: Path
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    device: Literal["cpu", "cuda"] = "cuda"
    output_dir: Path = Path("log")
    wandb_name: str | None = None
