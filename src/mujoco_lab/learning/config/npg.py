"""Natural policy gradient solver and step settings."""

from dataclasses import dataclass


@dataclass
class NPGConfig:
    """Natural gradient and empirical KL step limits."""

    max_kl: float = 0.01
    damping: float = 0.1
    cg_iterations: int = 10
    backtrack_steps: int = 10

    def __post_init__(self) -> None:
        if self.max_kl <= 0 or self.damping <= 0:
            raise ValueError("max_kl and damping must be positive")
        if self.cg_iterations <= 0 or self.backtrack_steps <= 0:
            raise ValueError("CG iterations and backtrack steps must be positive")
