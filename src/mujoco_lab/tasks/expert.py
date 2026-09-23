"""Action-producing expert contract shared by tasks and data collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Expert(ABC):
    """Produce physical actions while the caller owns environment stepping.

    Subclasses clear ``failed`` and ``failure_reason`` when reset for a new episode.
    """

    def __init__(self) -> None:
        self.failed = False
        self.failure_reason: str | None = None

    @abstractmethod
    def reset(self, initial_obs: Any = None, info: dict[str, Any] | None = None) -> None:
        """Reset expert state after the caller resets its environment."""

    @abstractmethod
    def act(self, obs: Any = None) -> np.ndarray:
        """Return one physical action without applying it."""
