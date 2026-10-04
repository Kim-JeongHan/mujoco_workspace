"""Demonstration episode storage and validation."""

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from mujoco_lab.learning.datasets.replay import replay_action_repeat


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

    def validate_training_data(self) -> tuple[int, int]:
        """Check observation and action arrays and return their feature dimensions."""
        if (
            self.states.ndim != 2
            or self.actions.ndim != 2
            or 0 in self.states.shape
            or 0 in self.actions.shape
            or len(self.states) != len(self.actions) + 1
        ):
            raise ValueError("states/actions must have shapes (T+1, S)/(T, A) with T,S,A > 0")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("states/actions must be finite")
        return self.states.shape[1], self.actions.shape[1]

    def check_physics_step_consistency(
        self, physics_steps_per_action: int, *, simulation_dt: float | None = None
    ) -> None:
        replay = self.metadata.get("replay") or {}
        repeat = replay_action_repeat(replay)
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


def save_episode(path: str | Path, episode: Episode) -> None:
    """Save one episode as compressed NPZ without replacing an existing file."""
    arrays: dict[str, np.ndarray | str] = {
        "states": episode.states,
        "actions": episode.actions,
        "metadata": json.dumps(episode.metadata),
    }
    for name in (
        "rewards",
        "terminated",
        "truncated",
        "qpos",
        "frame_times",
        "mocap_pos",
        "mocap_quat",
    ):
        value = getattr(episode, name)
        if value is not None:
            arrays[name] = value
    target = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **cast(dict[str, Any], arrays))
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_episode(path: str | Path) -> Episode:
    """Load an episode without allowing pickled objects in the NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        return Episode(
            states=data["states"],
            actions=data["actions"],
            rewards=data.get("rewards"),
            terminated=data.get("terminated"),
            truncated=data.get("truncated"),
            metadata=metadata,
            qpos=data.get("qpos"),
            frame_times=data.get("frame_times"),
            mocap_pos=data.get("mocap_pos"),
            mocap_quat=data.get("mocap_quat"),
        )


def load_episodes(data_dir: str | Path, *, success_only: bool = True) -> list[Episode]:
    """Load sorted top-level NPZ episodes with a common training shape.

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
        episode = load_episode(path)
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
