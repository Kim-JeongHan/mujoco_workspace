"""Demonstration episode storage and validation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Episode:
    states: np.ndarray  # (T + 1, state_dim)
    actions: np.ndarray  # (T, action_dim)

    rewards: np.ndarray | None = None  # (T,)
    terminated: np.ndarray | None = None  # (T,)
    truncated: np.ndarray | None = None  # (T,)

    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional full-scene playback data, separate from learning observations.
    qpos: np.ndarray | None = None  # (T + 1, model.nq)
    frame_times: np.ndarray | None = None  # (T + 1,)
    mocap_pos: np.ndarray | None = None  # (T + 1, model.nmocap, 3)
    mocap_quat: np.ndarray | None = None  # (T + 1, model.nmocap, 4)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def rotation_indices(self) -> list[int]:
        """Read rotation columns after training-data validation."""
        return self.metadata["observation"]["rotation_indices"]

    def validate_training_data(self) -> tuple[int, int]:
        """Check frame alignment and observation layout; return feature dimensions."""
        frame_count, state_dim = self.states.shape
        step_count, action_dim = self.actions.shape
        if frame_count != step_count + 1:
            raise ValueError("states must contain one more frame than actions")
        observation = self.metadata["observation"]
        if observation["frame_dim"] != state_dim:
            raise ValueError("Observation frame dimension does not match episode states")
        if any(not 0 <= index < state_dim for index in observation["rotation_indices"]):
            raise ValueError("Observation rotation_indices must be in range")
        return state_dim, action_dim

    def check_physics_step_consistency(
        self, physics_steps_per_action: int, *, simulation_dt: float | None = None
    ) -> None:
        replay = self.metadata["replay"]
        repeat = replay["physics_steps_per_action"]
        if repeat != physics_steps_per_action:
            raise ValueError(
                f"physics_steps_per_action={repeat} differs from config "
                f"{physics_steps_per_action}; recollect with "
                "--simulation-hz and --action-execution-hz matching the config"
            )

        if (
            simulation_dt is not None
            and "dt" in replay
            and not math.isclose(replay["dt"], simulation_dt, rel_tol=1e-10, abs_tol=1e-12)
        ):
            raise ValueError(
                f"Dataset dt={replay['dt']} differs from configured simulation dt={simulation_dt}; "
                "recollect with --simulation-hz matching the config"
            )

    def save(self, path: str | Path) -> None:
        """Save one episode as compressed NPZ without replacing an existing file."""
        arrays: dict[str, np.ndarray | str] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None:
                arrays[item.name] = json.dumps(value) if item.name == "metadata" else value
        target = Path(path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                np.savez_compressed(file, allow_pickle=False, **arrays)
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def load(path: str | Path) -> Episode:
        """Load an episode without allowing pickled objects in the NPZ file."""
        with np.load(path, allow_pickle=False) as data:
            arrays = dict(data)
            metadata = json.loads(str(arrays.pop("metadata").item()))
            return Episode(metadata=metadata, **arrays)

    @staticmethod
    def load_all(data_dir: str | Path, *, success_only: bool = True) -> list[Episode]:
        """Load sorted top-level NPZ episodes with validated observation metadata and shape.

        By default only episodes whose metadata has ``success is True`` are kept.
        Failed and unlabeled attempts remain available with ``success_only=False``.
        Optional rewards and replay arrays are left untouched.
        Loading and validation errors propagate unchanged.
        """
        directory = Path(data_dir)
        if not directory.is_dir():
            raise ValueError(f"Episode directory does not exist: {directory}")
        paths = sorted(path for path in directory.glob("*.npz") if path.is_file())
        if not paths:
            raise ValueError(f"No NPZ episodes found in {directory}")

        episodes: list[Episode] = []
        dimensions: tuple[int, int] | None = None
        for path in paths:
            episode = Episode.load(path)
            if success_only and episode.metadata.get("success") is not True:
                continue
            shape = episode.validate_training_data()
            if dimensions is None:
                dimensions = shape
            elif shape != dimensions:
                raise ValueError(f"feature dimensions {shape} differ from expected {dimensions}")
            episodes.append(episode)

        if not episodes:
            raise ValueError(f"No eligible episodes found in {directory}")
        return episodes
