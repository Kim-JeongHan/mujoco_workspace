"""Replay a recorded manipulation episode in the native MuJoCo viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import tyro
from dacite import from_dict

from mujoco_lab import (
    RobotSpec,
    Simulator,
    SimulatorManager,
    create_book_insertion,
    create_cube_stack,
)
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.learning.datasets.episode import Episode, load_episode
from mujoco_lab.learning.datasets.replay import model_signature
from mujoco_lab.utils.logger import Logger


@dataclass(frozen=True, kw_only=True)
class ReplayConfig:
    """Typed recorded scene settings used to reconstruct native replay."""

    scene: Literal["cube_stack", "book_insertion"]
    environment: Literal["table_shelf", "book_shelf"]
    robot: str
    robot_name: str
    dt: float
    physics_steps_per_action: int
    cubes: Literal[1, 2, 3, 4] | None = None
    book: Literal["small", "medium", "thick", "large"] | None = None
    model_sha256: str | None = None

    def create_scene(self) -> mujoco.MjSpec:
        """Build the recorded scene, requiring its scene-specific parameters."""
        if self.scene == "cube_stack" and self.environment == "table_shelf":
            if self.cubes is None:
                raise ValueError("Episode replay metadata is incomplete: missing cubes")
            return create_cube_stack(self.cubes)
        if self.scene == "book_insertion" and self.environment == "book_shelf":
            if self.book is None:
                raise ValueError("Episode replay metadata is incomplete: missing book")
            return create_book_insertion(self.book)
        raise ValueError("Unsupported replay scene; expected cube_stack or book_insertion")


@dataclass
class Config:
    """Choose an episode and playback speed."""

    path: Path  # Episode NPZ written by the collection CLI.
    speed: float = 1.0  # Recorded seconds per wall-clock second.


class EpisodeReplay:
    """Reconstruct a supported scene and apply its recorded geometry frames."""

    def __init__(self, episode: Episode) -> None:
        metadata = episode.metadata.get("replay")
        if (
            episode.qpos is None
            or episode.frame_times is None
            or episode.mocap_pos is None
            or episode.mocap_quat is None
            or not isinstance(metadata, dict)
        ):
            raise ValueError(
                "Episode has no replay data. Recollect with replay recording enabled; "
                "learning states are not MuJoCo qpos."
            )
        config = from_dict(data_class=ReplayConfig, data=metadata)
        self.simulator = Simulator(
            config.create_scene(),
            robots=[
                RobotSpec(
                    config.robot_name,
                    config.robot,
                    config=load_robot_config(config.robot),
                )
            ],
            dt=config.dt,
        )
        recorded_model = config.model_sha256
        if recorded_model is not None and recorded_model != model_signature(self.simulator.model):
            raise ValueError("Replay native joint coordinates require the recorded robot model")
        self.frame_count = len(episode) + 1
        self.frame_dt = self.simulator.dt * config.physics_steps_per_action
        self._qpos = episode.qpos
        self._mocap_pos = episode.mocap_pos
        self._mocap_quat = episode.mocap_quat
        self.frame_times = episode.frame_times.copy()

    def set_frame(self, index: int) -> None:
        """Restore a frame's qpos, mocap targets, and time without physics steps."""
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        model, data = self.simulator.model, self.simulator.data
        # Clear velocities, applied forces, and controls so GUI edits cannot
        # leave stale transient state while seeking backward or forward.
        mujoco.mj_resetData(model, data)
        data.qpos[:] = self._qpos[index]
        data.mocap_pos[:] = self._mocap_pos[index]
        data.mocap_quat[:] = self._mocap_quat[index]
        data.time = float(self.frame_times[index])
        mujoco.mj_forward(model, data)


def main() -> None:
    config = tyro.cli(Config, description="Replay a recorded manipulation episode")
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
            frame_times=replay.frame_times.tolist(),
        )
    finally:
        manager.remove_simulator("replay")


if __name__ == "__main__":
    main()
