"""Capture full-scene geometry for playback, independently of observations."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from mujoco_lab.learning.config.replay import BookReplayConfig, CubeStackReplayConfig
from mujoco_lab.simulation import Simulator


def model_signature(model: mujoco.MjModel) -> str:
    """Fingerprint the compiled model, including geometry and joint ordering."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer).hexdigest()


def capture_frame(simulator: Simulator) -> dict[str, np.ndarray]:
    """Copy one frame; learning observations cannot substitute for global qpos."""
    data = simulator.data
    return {
        "qpos": data.qpos.copy(),
        "frame_times": np.asarray(data.time),
        "mocap_pos": data.mocap_pos.copy(),
        "mocap_quat": data.mocap_quat.copy(),
    }


def capture_metadata(
    simulator: Simulator, config: CubeStackReplayConfig | BookReplayConfig
) -> dict[str, Any]:
    """Serialize scene settings and capture compiled-model information."""
    return {
        **asdict(config),
        "robot_name": next(iter(simulator.robots)),
        "dt": simulator.dt,
        "mujoco_version": mujoco.__version__,
        "model_sha256": model_signature(simulator.model),
    }
