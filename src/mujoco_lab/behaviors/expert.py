"""Action-producing expert contract shared by tasks and data collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Expert(ABC):
    """Produce physical actions while the caller owns environment stepping.

    Subclasses clear ``failed`` and ``failure_reason`` when reset for a new episode.
    """

    def __init__(self) -> None:
        self.failed = False
        self.failure_reason: str | None = None

    @abstractmethod
    def reset(self) -> None:
        """Reset expert state after the caller resets its environment."""

    @abstractmethod
    def act(self, *, dt: float = 0.0) -> np.ndarray | None:
        """Return the target dt seconds ahead without applying it.

        Use the environment's action_dt for its next action endpoint, or zero
        for a direct simulation callback requesting the current target.
        Return None when no new action is available after failure or completion;
        the caller retains its existing command while waiting for task success.
        """
