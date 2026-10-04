"""Replay a recorded cube-stacking episode in the native MuJoCo viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.learning.datasets.episode import Episode, load_episode
from mujoco_lab.learning.datasets.replay import (
    replay_action_repeat,
    visual_signature,
)
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
            "cubes",
            "robot",
            "robot_name",
            "dt",
            "visual_sha256",
        }
        if not required.issubset(metadata):
            raise ValueError("Episode replay metadata is incomplete")
        if metadata["scene"] != "cube_stack" or metadata["environment"] != "table_shelf":
            raise ValueError("Unsupported replay scene; expected a bundled cube-stack mount")
        self.simulator = Simulator(
            create_cube_stack(metadata["cubes"]),
            robots=[
                RobotSpec(
                    metadata["robot_name"],
                    metadata["robot"],
                    config=load_robot_config(metadata["robot"]),
                )
            ],
            dt=metadata["dt"],
        )
        model = self.simulator.model
        if visual_signature(model) != metadata["visual_sha256"]:
            raise ValueError(
                "Replay model differs from the recorded model. Use the same assets and layout."
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
    config = tyro.cli(Config, description="Replay a recorded cube-stacking episode")
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
