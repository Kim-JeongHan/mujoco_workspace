"""Behavior cloning training and checkpoint evaluation settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mujoco_lab.assets import RobotName
from mujoco_lab.learning.config.evaluation import EvaluationConfig
from mujoco_lab.learning.timing import physics_steps_per_action


@dataclass
class BCTrainConfig:
    """Offline BC defaults; dimensions come from the demonstration dataset."""

    # Data and scene.
    data_dir: Path = Path("data/forte_mixed_1")
    output_dir: Path = Path("log")
    robot: RobotName = "forte"

    # Policy architecture and observation/action timing.
    policy_type: Literal["mse", "flow", "diffusion"] = "mse"
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    obs_horizon: int = 2  # Observation frames per policy input.
    chunk_size: int = 16  # Actions predicted per policy output.
    execution_horizon: int = 4  # Actions before replanning; 100 Hz / 4 = 25 Hz inference.
    simulation_hz: float = 500.0  # Physics and controller evaluations per simulated second.
    action_execution_hz: float = 100.0  # 500 Hz / 100 Hz = 5 physics steps/action.
    sampling_num_steps: int = 20  # Euler steps for Flow or DDIM steps for Diffusion.
    flow_time_embed_dim: int = 256  # Sinusoidal time feature width.
    diffusion_down_dims: tuple[int, ...] = (128, 256, 512)
    diffusion_time_embed_dim: int = 256
    diffusion_num_train_timesteps: int = 100  # Training noise levels, separate from sampling steps.
    diffusion_beta_schedule: Literal["linear", "squaredcos_cap_v2"] = "squaredcos_cap_v2"

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
    run_name: str | None = "forte-mse-cube1"

    @property
    def simulation_dt(self) -> float:
        return 1.0 / self.simulation_hz

    @property
    def physics_steps_per_action(self) -> int:
        return physics_steps_per_action(self.simulation_hz, self.action_execution_hz)


@dataclass
class BCEvalConfig:
    """Evaluate one saved BC policy in the recorded physical task environment."""

    checkpoint: Path
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: Literal["cpu", "cuda"] = "cuda"
    output_dir: Path = Path("log")
    run_name: str | None = None
