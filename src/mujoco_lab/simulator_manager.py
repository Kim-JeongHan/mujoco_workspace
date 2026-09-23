"""Process-wide resources shared by Simulator instances."""

from __future__ import annotations

import time
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from math import isfinite
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import mujoco

from mujoco_lab.rendering.video import VideoRecorder
from mujoco_lab.utils import Logger

if TYPE_CHECKING:
    from mujoco_lab.simulation import Simulator

SceneDraw = Callable[[mujoco.MjvScene, mujoco.MjData], None]

__all__ = ["SimulatorManager"]


def _copy_camera(source: mujoco.MjvCamera) -> mujoco.MjvCamera:
    """Snapshot the viewer camera for rendering on another scene."""
    camera = mujoco.MjvCamera()
    camera.type = source.type
    camera.fixedcamid = source.fixedcamid
    camera.trackbodyid = source.trackbodyid
    camera.lookat[:] = source.lookat
    camera.distance = source.distance
    camera.azimuth = source.azimuth
    camera.elevation = source.elevation
    return camera


class SimulatorManager:
    """Own resources shared by Simulator instances."""

    _instance: ClassVar[SimulatorManager | None] = None
    logger: Logger
    simulators: dict[str, Simulator]

    def __init__(self) -> None:
        self.logger = Logger()
        self.simulators = {}

    @classmethod
    def get_instance(cls) -> SimulatorManager:
        """Return the lazily created process-wide manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_simulator(self, name: str, simulator: Simulator) -> None:
        """Register a Simulator under an unused name."""
        if name in self.simulators:
            raise KeyError(name)
        self.simulators[name] = simulator

    def remove_simulator(self, name: str) -> Simulator:
        """Remove and return the Simulator registered under a name."""
        return self.simulators.pop(name)

    def replace_simulator(self, name: str, simulator: Simulator) -> Simulator:
        """Replace a registered Simulator and return the previous one."""
        previous = self.simulators[name]
        self.simulators[name] = simulator
        return previous

    def show(
        self,
        name: str,
        draw: SceneDraw | None = None,
        camera: mujoco.MjvCamera | None = None,
        *,
        video: str | Path | None = None,
        fps: float = 30,
        width: int = 640,
        height: int = 480,
        steps: int | None = None,
    ) -> Path | None:
        """Step in the native viewer, drawing owned user geoms each tick.

        An optional camera sets the initial view; users can move it afterward.
        """
        if video is not None:
            return self._show_recording(name, video, draw, camera, fps, width, height, steps)
        simulator = self.simulators[name]

        import mujoco.viewer

        from mujoco_lab.simulation import SimulatorState

        simulator._state.transition(SimulatorState.VIEWING)
        simulator._stop_requested = False
        # launch_passive calls mj_forward on the live data before its
        # private render copy exists. Restore the exact pre-launch data
        # so entering the GUI does not change the next control sample.
        launch_data = mujoco.MjData(simulator.model)
        mujoco.mj_copyData(launch_data, simulator.model, simulator.data)
        try:
            viewer = mujoco.viewer.launch_passive(simulator.model, simulator.data)
        finally:
            mujoco.mj_copyData(simulator.data, simulator.model, launch_data)
        try:
            if camera is not None:
                with viewer.lock():
                    viewer.cam.type = camera.type
                    viewer.cam.fixedcamid = camera.fixedcamid
                    viewer.cam.trackbodyid = camera.trackbodyid
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance = camera.distance
                    viewer.cam.azimuth = camera.azimuth
                    viewer.cam.elevation = camera.elevation
            while not simulator._stop_requested and viewer.is_running():
                started = time.monotonic()
                with viewer.lock():
                    simulator.physics_step()
                    if draw is not None:
                        viewer.user_scn.ngeom = 0
                        draw(viewer.user_scn, simulator.data)
                try:
                    viewer.sync(state_only=True)
                finally:
                    # Timing is a Simulator construction setting. Ignore
                    # a passive Physics-panel edit so all clocks align.
                    with viewer.lock():
                        simulator.model.opt.timestep = simulator.dt
                remaining = simulator.dt - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            viewer.close()
            # close() requests exit but does not join the daemon render
            # thread. Wait for its native owner before returning so that
            # interpreter shutdown cannot race viewer destruction.
            simulate_ref = getattr(viewer, "_sim", None)
            if simulate_ref is not None:
                while simulate_ref() is not None:
                    time.sleep(0.001)
        simulator._state.transition(SimulatorState.IDLE)
        simulator._stop_requested = False

    def _show_recording(
        self,
        name: str,
        output: str | Path,
        draw: SceneDraw | None,
        camera: mujoco.MjvCamera | None,
        fps: float,
        width: int,
        height: int,
        steps: int | None,
    ) -> Path | None:
        """Preview a live camera, then use Space to start and R to finish MP4 recording."""
        if steps is not None:
            steps = index(steps)
            if steps < 0:
                raise ValueError("steps must be zero or greater")
        simulator = self.simulators[name]
        import mujoco.viewer

        from mujoco_lab.simulation import SimulatorState

        keys: deque[int] = deque()
        simulator._state.transition(SimulatorState.VIEWING)
        simulator._stop_requested = False
        viewer = None
        try:
            launch_data = mujoco.MjData(simulator.model)
            mujoco.mj_copyData(launch_data, simulator.model, simulator.data)
            try:
                viewer = mujoco.viewer.launch_passive(
                    simulator.model, simulator.data, key_callback=keys.append
                )
            finally:
                mujoco.mj_copyData(simulator.data, simulator.model, launch_data)
            with viewer.lock():
                if camera is not None:
                    viewer.cam.type = camera.type
                    viewer.cam.fixedcamid = camera.fixedcamid
                    viewer.cam.trackbodyid = camera.trackbodyid
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance = camera.distance
                    viewer.cam.azimuth = camera.azimuth
                    viewer.cam.elevation = camera.elevation
            self.logger.info(
                "Adjust the camera, then press Space to start recording; press R to finish"
            )
            recorder = None
            completed = 0
            with ExitStack() as stack:
                while not simulator._stop_requested and viewer.is_running():
                    started = time.monotonic()
                    stop_recording = False
                    while keys:
                        key = keys.popleft()
                        if key == 32 and recorder is None:  # Space starts from the current state.
                            recorder = stack.enter_context(
                                VideoRecorder(
                                    simulator,
                                    output,
                                    fps=fps,
                                    width=width,
                                    height=height,
                                    draw=draw,
                                )
                            )
                            with viewer.lock():
                                recorder.record_initial(_copy_camera(viewer.cam))
                        elif key == 82 and recorder is not None:  # R finishes the MP4.
                            stop_recording = True
                    if stop_recording or (
                        recorder is not None and steps is not None and completed >= steps
                    ):
                        break
                    with viewer.lock():
                        if recorder is not None:
                            simulator.physics_step()
                            completed += 1
                            recorder.record_due(_copy_camera(viewer.cam))
                        if draw is not None:
                            viewer.user_scn.ngeom = 0
                            draw(viewer.user_scn, simulator.data)
                    try:
                        viewer.sync(state_only=True)
                    finally:
                        with viewer.lock():
                            simulator.model.opt.timestep = simulator.dt
                    remaining = (simulator.dt if recorder is not None else 1 / 60) - (
                        time.monotonic() - started
                    )
                    if remaining > 0:
                        time.sleep(remaining)
            return None if recorder is None else Path(output).expanduser().resolve()
        finally:
            if viewer is not None:
                viewer.close()
                simulate_ref = getattr(viewer, "_sim", None)
                if simulate_ref is not None:
                    while simulate_ref() is not None:
                        time.sleep(0.001)
            simulator._state.transition(SimulatorState.IDLE)
            simulator._stop_requested = False

    def save_video(
        self,
        name: str,
        steps: int,
        output: str | Path,
        camera: mujoco.MjvCamera | None = None,
        *,
        fps: float = 30,
        width: int = 640,
        height: int = 480,
        draw: SceneDraw | None = None,
    ) -> Path:
        """Run physics once per tick while streaming a copied-data MP4."""
        steps = index(steps)
        if steps < 0:
            raise ValueError("steps must be zero or greater")
        simulator = self.simulators[name]
        from mujoco_lab.simulation import SimulatorState

        simulator._state.transition(SimulatorState.RUNNING)
        simulator._stop_requested = False
        try:
            with VideoRecorder(
                simulator, output, fps=fps, width=width, height=height, draw=draw
            ) as recorder:
                recorder.record_initial(camera)
                for _ in range(steps):
                    if simulator._stop_requested:
                        break
                    simulator.physics_step()
                    recorder.record_due(camera)
            return Path(output).expanduser().resolve()
        finally:
            simulator._state.transition(SimulatorState.IDLE)
            simulator._stop_requested = False

    def show_replay(
        self,
        name: str,
        frame_count: int,
        frame_dt: float,
        set_frame: Callable[[int], None],
        *,
        speed: float = 1.0,
        camera: mujoco.MjvCamera | None = None,
        frame_times: Sequence[float] | None = None,
    ) -> None:
        """Display recorded frames without advancing physics or evaluating control.

        ``set_frame`` restores scene state and calls ``mj_forward``. Space toggles
        playback, Left/Right pause and step one frame, and R or Home rewinds.
        Playback pauses on the final frame; Space there restarts from frame zero.
        Frames may be skipped to keep wall-clock timing at the 60 Hz display rate.
        ``frame_times`` uses recorded timestamps when action intervals vary.
        """
        simulator = self.simulators[name]
        frame_count = index(frame_count)
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if not isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError("frame_dt must be finite and positive")
        if not isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        offsets = None
        if frame_times is not None:
            if len(frame_times) != frame_count or not all(isfinite(value) for value in frame_times):
                raise ValueError("frame_times must contain one finite time per frame")
            offsets = [float(value) - float(frame_times[0]) for value in frame_times]
            if any(later <= earlier for earlier, later in zip(offsets, offsets[1:], strict=False)):
                raise ValueError("frame_times must increase strictly")

        import mujoco.viewer

        from mujoco_lab.simulation import SimulatorState

        # The viewer invokes callbacks on its render thread. Consume input only
        # on this thread, alongside state restoration under the viewer lock.
        keys: deque[int] = deque()
        simulator._state.transition(SimulatorState.VIEWING)
        simulator._stop_requested = False
        viewer = None
        try:
            set_frame(0)
            viewer = mujoco.viewer.launch_passive(
                simulator.model, simulator.data, key_callback=keys.append
            )
            if camera is not None:
                with viewer.lock():
                    viewer.cam.type = camera.type
                    viewer.cam.fixedcamid = camera.fixedcamid
                    viewer.cam.trackbodyid = camera.trackbodyid
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance = camera.distance
                    viewer.cam.azimuth = camera.azimuth
                    viewer.cam.elevation = camera.elevation
            frame = 0
            playhead = 0.0
            playing = True
            previous = time.monotonic()
            while not simulator._stop_requested and viewer.is_running():
                started = time.monotonic()
                if playing:
                    playhead += (started - previous) * speed
                    frame = (
                        min(int(playhead / frame_dt), frame_count - 1)
                        if offsets is None
                        else min(bisect_right(offsets, playhead) - 1, frame_count - 1)
                    )
                    if frame == frame_count - 1:
                        playing = False
                previous = started
                while keys:
                    key = keys.popleft()
                    if key == 32:  # Space
                        if not playing and frame == frame_count - 1:
                            frame = 0
                        playing = not playing
                    elif key in (262, 263):  # GLFW Right / Left
                        playing = False
                        frame = max(0, min(frame + (1 if key == 262 else -1), frame_count - 1))
                    elif key in (82, 268):  # R / GLFW Home
                        frame = 0
                    else:
                        continue
                    playhead = frame * frame_dt if offsets is None else offsets[frame]
                with viewer.lock():
                    set_frame(frame)
                viewer.sync(state_only=True)
                remaining = 1 / 60 - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if viewer is not None:
                viewer.close()
                simulate_ref = getattr(viewer, "_sim", None)
                if simulate_ref is not None:
                    while simulate_ref() is not None:
                        time.sleep(0.001)
            simulator._state.transition(SimulatorState.IDLE)
            simulator._stop_requested = False

    def save_frame(
        self,
        name: str,
        output: str | Path = "frame.png",
        draw: SceneDraw | None = None,
        camera: mujoco.MjvCamera | None = None,
    ) -> Path:
        """Render copied data with optional extra geoms and camera framing."""
        simulator = self.simulators[name]

        from PIL import Image

        from mujoco_lab.rendering.annotations import annotate
        from mujoco_lab.simulation import SimulatorState

        path = Path(output).expanduser().resolve()
        simulator._state.transition(SimulatorState.RENDERING)
        scratch = mujoco.MjData(simulator.model)
        mujoco.mj_copyData(scratch, simulator.model, simulator.data)
        mujoco.mj_forward(simulator.model, scratch)
        with mujoco.Renderer(simulator.model, height=480, width=640) as renderer:
            if camera is None:
                renderer.update_scene(scratch)
            else:
                renderer.update_scene(scratch, camera=camera)
            annotate(renderer.scene, simulator, scratch)
            if draw is not None:
                draw(renderer.scene, scratch)
            pixels = renderer.render()
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(path, format="PNG")
        simulator._state.transition(SimulatorState.IDLE)
        return path
