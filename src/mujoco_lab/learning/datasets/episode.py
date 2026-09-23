import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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
