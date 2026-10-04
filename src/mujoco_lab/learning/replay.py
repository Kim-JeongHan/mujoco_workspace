"""Replay a recorded manipulation episode in the native MuJoCo viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

from mujoco_lab import (
    RobotSpec,
    Simulator,
    SimulatorManager,
    create_book_insertion,
    create_cube_stack,
)
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.learning.datasets.episode import Episode, load_episode
from mujoco_lab.learning.datasets.replay import replay_action_repeat
from mujoco_lab.utils.logger import Logger


@dataclass
class Config:
    """Choose an episode and playback speed."""

    path: Path  # Episode NPZ written by the collection CLI.
    speed: float = 1.0  # Recorded seconds per wall-clock second.


class EpisodeReplay:
    """Reconstruct a supported scene and apply its recorded geometry frames."""

    def __init__(self, episode: Episode) -> None:
        metadata = episode.metadata.get("replay")
        if episode.qpos is None or not isinstance(metadata, dict):
            raise ValueError(
                "Episode has no replay data. Recollect with replay recording enabled; "
                "learning states are not MuJoCo qpos."
            )
        required = {
            "scene",
            "environment",
            "robot",
            "robot_name",
            "dt",
        }
        if not required.issubset(metadata):
            raise ValueError("Episode replay metadata is incomplete")
        if metadata["scene"] == "cube_stack" and metadata["environment"] == "table_shelf":
            if "cubes" not in metadata:
                raise ValueError("Episode replay metadata is incomplete: missing cubes")
            scene = create_cube_stack(metadata["cubes"])
        elif metadata["scene"] == "book_insertion" and metadata["environment"] == "book_shelf":
            if "book" not in metadata:
                raise ValueError("Episode replay metadata is incomplete: missing book")
            scene = create_book_insertion(metadata["book"])
        else:
            raise ValueError("Unsupported replay scene; expected cube_stack or book_insertion")
        self.simulator = Simulator(
            scene,
            robots=[
                RobotSpec(
                    metadata["robot_name"],
                    metadata["robot"],
                    config=load_robot_config(metadata["robot"]),
                )
            ],
            dt=metadata["dt"],
        )
        self.frame_count = len(episode) + 1
        action_repeat = replay_action_repeat(metadata)
        self.frame_dt = self.simulator.dt * action_repeat
        self._frames: dict[str, np.ndarray] = {
            name: getattr(episode, name)
            for name in ("qpos", "frame_times", "mocap_pos", "mocap_quat")
        }
        self.frame_times = self._frames["frame_times"].copy()

    def set_frame(self, index: int) -> None:
        """Restore a frame's qpos, mocap targets, and time without physics steps."""
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        model, data = self.simulator.model, self.simulator.data
        # Clear velocities, applied forces, and controls so GUI edits cannot
        # leave stale transient state while seeking backward or forward.
        mujoco.mj_resetData(model, data)
        data.qpos[:] = self._frames["qpos"][index]
        data.mocap_pos[:] = self._frames["mocap_pos"][index]
        data.mocap_quat[:] = self._frames["mocap_quat"][index]
        data.time = float(self._frames["frame_times"][index])
        mujoco.mj_forward(model, data)


def main() -> None:
    config = tyro.cli(Config, description="Replay a recorded manipulation episode")
    if not np.isfinite(config.speed) or config.speed <= 0:
        raise ValueError("speed must be finite and positive")
    replay = EpisodeReplay(load_episode(config.path))
    manager = SimulatorManager()
    manager.add_simulator("replay", replay.simulator)
    Logger().info("Space: play/pause; Left/Right: step; R/Home: rewind; close the viewer to exit.")
    try:
        manager.show_replay(
            "replay",
            replay.frame_count,
            replay.frame_dt,
            replay.set_frame,
            speed=config.speed,
            frame_times=replay.frame_times,
        )
    finally:
        manager.remove_simulator("replay")


if __name__ == "__main__":
    main()
