"""Owned joint state snapshots for control and observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointState:
    """One robot's state in controller joint order and SI units.

    RobotState.snapshot() supplies owned arrays that remain stable across later
    reads or steps. Arrays are mutable copies; frozen prevents field reassignment.
    Bias forces include gravity and velocity-dependent terms, matching qfrc_bias.
    """

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    bias_forces: np.ndarray
