"""Diffusion scheduler settings."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DiffusionSchedulerConfig:
    """DDIM settings saved with the policy for reproducible inference."""

    num_train_timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "squaredcos_cap_v2"
    trained_betas: list[float] | None = None
    clip_sample: bool = False
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    clip_sample_range: float = 1.0
    sample_max_value: float = 1.0
    timestep_spacing: Literal["leading", "trailing", "linspace"] = "leading"
    rescale_betas_zero_snr: bool = False
