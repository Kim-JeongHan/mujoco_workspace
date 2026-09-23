"""Camera helpers for MuJoCo rendering."""

import mujoco
import numpy as np


def create_free_camera(
    *,
    lookat: np.ndarray,
    azimuth: float = 60.0,
    elevation: float = -25.0,
    distance: float = 1.25,
) -> mujoco.MjvCamera:
    """Create a free camera aimed at the given point."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    return camera
