"""Behavior cloning training and checkpoint evaluation settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mujoco_lab.assets import RobotName
from mujoco_lab.learning.config.diffusion import DiffusionSchedulerConfig
from mujoco_lab.learning.config.evaluation import EvaluationConfig
from mujoco_lab.learning.timing import physics_steps_per_action


@dataclass
class BCTrainConfig:
    """Shared offline BC data, optimization, timing, and evaluation settings."""

    policy_type: Literal["mse", "flow", "diffusion"] = "mse"

    # Data and scene.
    data_dir: Path = Path("data/forte_mixed_1")
    split_file: Path | None = None  # Reuse ordered seed assignments from a collection.
    output_dir: Path = Path("log")
    robot: RobotName = "forte"

    # Observation/action timing.
    obs_horizon: int = 2  # Observation frames per policy input.
    chunk_size: int = 16  # Actions predicted per policy output.
    execution_horizon: int = 4  # Actions before replanning; 100 Hz / 4 = 25 Hz inference.
    simulation_hz: float = 500.0  # Physics and controller evaluations per simulated second.
    action_execution_hz: float = 100.0  # 500 Hz / 100 Hz = 5 physics steps/action.

    # Optimization.
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    ema_decay: float | None = 0.999  # None disables parameter averaging.
    num_epochs: int = 400
    num_workers: int = 0

    # Evaluation and logging.
    eval_interval_steps: int = 30_000  # Optimizer steps; nonpositive disables evaluation.
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    log_interval_steps: int = 100  # Optimizer steps between local and W&B metric logs.

    # Episode-level dataset splits.
    validation_ratio: float = 0.1
    test_ratio: float = 0.0

    # Reproducibility and run identity.
    seed: int = 42
    run_name: str | None = None

    @property
    def simulation_dt(self) -> float:
        return 1.0 / self.simulation_hz

    @property
    def physics_steps_per_action(self) -> int:
        return physics_steps_per_action(self.simulation_hz, self.action_execution_hz)


@dataclass
class MSEBCTrainConfig(BCTrainConfig):
    """Train an MLP action regressor on demonstration chunks."""

    hidden_dims: tuple[int, ...] = (512, 512, 512)


@dataclass
class FlowBCTrainConfig(BCTrainConfig):
    """Train conditional flow matching on demonstration chunks."""

    sampling_num_steps: int = 20  # Euler integration steps.
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    time_embed_dim: int = 256


@dataclass
class DiffusionBCTrainConfig(BCTrainConfig):
    """Train a temporal U-Net with the configured diffusion noise schedule."""

    sampling_num_steps: int = 20  # DDIM sampling steps, independent of training noise levels.
    down_dims: tuple[int, ...] = (128, 256, 512)
    time_embed_dim: int = 256
    scheduler: DiffusionSchedulerConfig = field(default_factory=DiffusionSchedulerConfig)


type BCPolicyTrainConfig = MSEBCTrainConfig | FlowBCTrainConfig | DiffusionBCTrainConfig


@dataclass
class BCEvalConfig:
    """Evaluate one saved BC policy in the recorded physical task environment."""

    checkpoint: Path
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: Literal["cpu", "cuda"] = "cuda"
    output_dir: Path = Path("log")
    run_name: str | None = None
