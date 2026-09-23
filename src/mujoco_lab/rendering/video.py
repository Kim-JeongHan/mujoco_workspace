"""Stream copied MuJoCo frames to an H.264 MP4 file."""

from __future__ import annotations

import subprocess
from math import ceil, isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from mujoco_lab.simulation import Simulator

    SceneDraw = Callable[[mujoco.MjvScene, mujoco.MjData], None]


class VideoWriter:
    """Write RGB frames to FFmpeg without retaining the full video in memory."""

    def __init__(self, output: str | Path, fps: float, width: int, height: int) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("video width and height must be positive even integers")
        self.path = Path(output).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, pixels: np.ndarray) -> None:
        """Append one RGB frame to the encoder."""
        if pixels.shape != (self.height, self.width, 3) or pixels.dtype != np.uint8:
            raise ValueError("video frame must be uint8 RGB at the configured size")
        self._process.stdin.write(pixels.tobytes())

    def close(self) -> Path:
        """Finish MP4 metadata and report encoder errors."""
        broken_pipe = False
        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                broken_pipe = True
        error = self._process.stderr.read().decode(errors="replace").strip()
        if self._process.wait() != 0 or broken_pipe:
            raise RuntimeError(f"FFmpeg could not write {self.path}: {error}")
        return self.path


class VideoRecorder:
    """Sample simulator time and render from a private MuJoCo data copy."""

    def __init__(
        self,
        simulator: Simulator,
        output: str | Path,
        *,
        fps: float = 30,
        width: int = 640,
        height: int = 480,
        draw: SceneDraw | None = None,
    ) -> None:
        self.simulator = simulator
        self.output = output
        self.fps = fps
        self.width, self.height = width, height
        self.draw = draw
        self._scratch = mujoco.MjData(simulator.model)
        self._start_time = float(simulator.data.time)
        self._frames = 0

    def __enter__(self) -> VideoRecorder:
        global_vis = self.simulator.model.vis.global_
        self._offscreen_size = (global_vis.offwidth, global_vis.offheight)
        global_vis.offwidth = max(global_vis.offwidth, self.width)
        global_vis.offheight = max(global_vis.offheight, self.height)
        try:
            self._renderer = mujoco.Renderer(
                self.simulator.model, height=self.height, width=self.width
            )
        except BaseException:
            global_vis.offwidth, global_vis.offheight = self._offscreen_size
            raise
        try:
            self._writer = VideoWriter(self.output, self.fps, self.width, self.height)
        except BaseException:
            try:
                self._renderer.close()
            finally:
                global_vis.offwidth, global_vis.offheight = self._offscreen_size
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self._writer.close()
        finally:
            try:
                self._renderer.close()
            finally:
                (
                    self.simulator.model.vis.global_.offwidth,
                    self.simulator.model.vis.global_.offheight,
                ) = self._offscreen_size

    def _render(self, camera: mujoco.MjvCamera | None) -> np.ndarray:
        simulator = self.simulator
        mujoco.mj_copyData(self._scratch, simulator.model, simulator.data)
        mujoco.mj_forward(simulator.model, self._scratch)
        if camera is None:
            self._renderer.update_scene(self._scratch)
        else:
            self._renderer.update_scene(self._scratch, camera=camera)
        if self.draw is not None:
            self.draw(self._renderer.scene, self._scratch)
        return self._renderer.render()

    def record_initial(self, camera: mujoco.MjvCamera | None) -> None:
        """Record the unchanged start state at video time zero."""
        self._writer.write(self._render(camera))
        self._frames = 1

    def record_due(self, camera: mujoco.MjvCamera | None) -> None:
        """Write frames due before the current simulated endpoint."""
        elapsed = float(self.simulator.data.time) - self._start_time
        due = ceil(elapsed * self.fps - 1e-9)
        if due <= self._frames:
            return
        pixels = self._render(camera)
        while self._frames < due:
            self._writer.write(pixels)
            self._frames += 1
