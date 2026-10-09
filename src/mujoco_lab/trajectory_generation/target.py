"""Desired joint or world-frame task state supplied to a controller."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, init=False)
class ControlTarget:
    """Owned position, velocity, and acceleration arrays; omitted derivatives are zero.

    Array-like inputs are copied into floating-point arrays at construction.
    Omitted or ``None`` derivatives become zero arrays matching the position shape.
    """

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __init__(
        self,
        position: ArrayLike,
        velocity: ArrayLike | None = None,
        acceleration: ArrayLike | None = None,
    ) -> None:
        position_array = np.array(position, dtype=float, copy=True)
        object.__setattr__(self, "position", position_array)
        for name, value in (("velocity", velocity), ("acceleration", acceleration)):
            array = (
                np.zeros_like(position_array)
                if value is None
                else np.array(value, dtype=float, copy=True)
            )
            if array.shape != position_array.shape:
                raise ValueError(f"Target {name} must match position shape")
            object.__setattr__(self, name, array)
