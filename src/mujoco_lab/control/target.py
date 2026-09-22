"""Desired joint or world-frame task state supplied to a controller."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlTarget:
    """Position with optional velocity and acceleration; omitted derivatives are zero."""

    position: np.ndarray
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "acceleration"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, np.array(value, dtype=float, copy=True))
