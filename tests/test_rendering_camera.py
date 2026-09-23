"""Free camera construction for rendering."""

import mujoco
import numpy as np

from mujoco_lab.rendering.camera import create_free_camera


def test_create_free_camera_defaults_and_copies_lookat():
    lookat = np.array([0.1, 0.2, 0.3])
    camera = create_free_camera(lookat=lookat)

    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    np.testing.assert_array_equal(camera.lookat, lookat)
    assert (camera.azimuth, camera.elevation, camera.distance) == (60.0, -25.0, 1.25)

    lookat[:] = 0
    np.testing.assert_array_equal(camera.lookat, [0.1, 0.2, 0.3])


def test_create_free_camera_custom_view():
    camera = create_free_camera(
        lookat=np.array([-0.2, 0.4, 0.6]), azimuth=90.0, elevation=-15.0, distance=2.0
    )

    np.testing.assert_array_equal(camera.lookat, [-0.2, 0.4, 0.6])
    assert (camera.azimuth, camera.elevation, camera.distance) == (90.0, -15.0, 2.0)
