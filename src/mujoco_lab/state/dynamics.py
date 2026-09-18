"""Read frame positions, Jacobians, and mass matrices from MuJoCo."""

import mujoco
import numpy as np


class Dynamics:
    """Provide world-frame dynamics using reusable NumPy buffers owned by this object.

    Public frame names map to MJCF sites. Jacobian rows are ordered as linear
    then angular velocity; columns follow MuJoCo's generalized velocity order.
    Getters use the current derived data without calling mj_forward or stepping.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.nv = model.nv
        self._jacobian = np.zeros((6, self.nv))
        self._mass = np.zeros((self.nv, self.nv))

    def _site_id(self, site: str) -> int:
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if index < 0:
            raise ValueError(f"Model has no site named {site!r}")
        return index

    def get_frame_position(self, frame: str) -> np.ndarray:
        """Borrow the site origin in world coordinates, shape (3,), in meters."""
        return self.data.site_xpos[self._site_id(frame)]

    def get_jacobian(self, frame: str) -> np.ndarray:
        """Return a (6, nv) buffer overwritten by the next call for any frame."""
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._jacobian[:3],
            self._jacobian[3:],
            self._site_id(frame),
        )
        return self._jacobian

    def get_mass_matrix(self) -> np.ndarray:
        """Return an (nv, nv) buffer overwritten by the next mass-matrix call."""
        mujoco.mj_fullM(self.model, self.data, self._mass)
        return self._mass
