"""Process-wide resources shared by Simulator instances."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import mujoco

from mujoco_lab.utils import Logger

if TYPE_CHECKING:
    from mujoco_lab.simulation import Simulator

SceneDraw = Callable[[mujoco.MjvScene, mujoco.MjData], None]

__all__ = ["SimulatorManager"]


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
    ) -> None:
        """Step in the native viewer, drawing owned user geoms each tick.

        An optional camera sets the initial view; users can move it afterward.
        """
        simulator = self.simulators[name]

        import mujoco.viewer

        from mujoco_lab.simulation import SimulatorState

        simulator._require_idle()
        simulator._stop_requested = False
        simulator._state.transition(SimulatorState.VIEWING)
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
        simulator._require_idle()
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
