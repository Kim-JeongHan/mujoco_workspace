"""Robot-local state and dynamics access over a shared MuJoCo model/data pair."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mujoco_lab.state.joint_state import JointState
from mujoco_lab.utils import Transform


def _descendants(model, root_id: int) -> set[int]:
    result = {root_id}
    for body_id in range(root_id + 1, model.nbody):
        if model.body_parentid[body_id] in result:
            result.add(body_id)
    return result


class RobotState:
    """Read robot state and dynamics, and solve IK without owning a physics loop.

    The model/data references belong to Simulator. Joint and frame bindings stay
    fixed, while reads use the current shared data. Getters never refresh shared
    physics or advance time. IK evaluates kinematics on a private data copy.
    Dynamics retain MuJoCo's most recent evaluation phase.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        name: str,
        prefix: str,
        root_name: str,
        joint_names: tuple[str, ...],
        site_names: tuple[str, ...],
    ) -> None:
        self.model, self.data = model, data
        self.name, self.prefix = name, prefix
        self.joint_names = joint_names
        self.root_body_id = model.body(prefix + root_name).id
        bodies = _descendants(model, self.root_body_id)
        self.joint_ids = [model.joint(prefix + name).id for name in joint_names]
        if any(model.jnt_bodyid[j] not in bodies for j in self.joint_ids):
            raise ValueError("Robot joints must belong to its root body subtree")
        qpos, dofs = [], []
        # Convert enum keys to match NumPy joint-type values.
        widths = {
            int(mujoco.mjtJoint.mjJNT_FREE): (7, 6),
            int(mujoco.mjtJoint.mjJNT_BALL): (4, 3),
            int(mujoco.mjtJoint.mjJNT_SLIDE): (1, 1),
            int(mujoco.mjtJoint.mjJNT_HINGE): (1, 1),
        }
        for joint_id in self.joint_ids:
            nq, nv = widths[model.jnt_type[joint_id]]
            qadr, dadr = model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]
            qpos.extend(range(qadr, qadr + nq))
            dofs.extend(range(dadr, dadr + nv))
        self.qpos_indices, self.dof_indices = qpos, dofs
        self.nq, self.nv = len(qpos), len(dofs)
        self._sites = {name: model.site(prefix + name).id for name in site_names}
        self._full_jacobian = np.zeros((6, model.nv))
        self._full_mass = np.zeros((model.nv, model.nv))
        self._jacobian = np.zeros((6, self.nv))
        self._mass = np.zeros((self.nv, self.nv))
        self._ik_data = None

    def snapshot(self) -> JointState:
        """Copy current joint fields into an independent snapshot, without forward."""
        return JointState(
            self.data.time,
            self.data.qpos[self.qpos_indices],
            self.data.qvel[self.dof_indices],
            self.data.qfrc_bias[self.dof_indices],
        )

    def site_id(self, frame: str) -> int:
        """Resolve a local site name belonging to this robot."""
        try:
            return self._sites[frame]
        except KeyError:
            raise ValueError(f"Robot {self.name!r} has no site named {frame!r}") from None

    def get_frame_position(self, frame: str) -> np.ndarray:
        """Borrow a site position in world coordinates, in meters."""
        return self.data.site_xpos[self.site_id(frame)]

    def get_jacobian(self, frame: str) -> np.ndarray:
        """Borrow the local (6, nv) geometric Jacobian, linear rows first."""

        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._full_jacobian[:3],
            self._full_jacobian[3:],
            self.site_id(frame),
        )
        self._jacobian[:] = self._full_jacobian[:, self.dof_indices]
        return self._jacobian

    def get_mass_matrix(self) -> np.ndarray:
        """Borrow the robot DOF block of the full generalized mass matrix."""
        mujoco.mj_fullM(self.model, self.data, self._full_mass)
        self._mass[:] = self._full_mass[np.ix_(self.dof_indices, self.dof_indices)]
        return self._mass

    def solve_ik(
        self,
        target_pose: Transform,
        *,
        frame: str,
        seed: ArrayLike | None = None,
    ) -> np.ndarray:
        """Solve a world-space site pose without modifying shared simulation data.

        Only this robot's hinge/slide joints on the site's ancestor chain are
        optimized; descendant gripper joints and other robots remain fixed.
        Returned positions and an optional seed follow their joint_names order.
        Limited joints keep a 1e-4 margin; continuous joints are unbounded.

        The supplied seed is tried first, then the current selected positions.
        Rotation-vector errors have weight 0.18 in the least-squares residual,
        with acceptance at 8 mm position error and 0.06 weighted rotation error
        (1/3 rad). Raise ValueError if neither attempt meets these tolerances.
        The result is not collision-checked.
        """
        site = self.site_id(frame)
        ancestors = set()
        body = self.model.site_bodyid[site]
        while body:
            ancestors.add(body)
            body = self.model.body_parentid[body]
        joints = [joint for joint in self.joint_ids if self.model.jnt_bodyid[joint] in ancestors]
        if not joints:
            raise ValueError(f"Frame {frame!r} has no movable joints belonging to this robot")
        if any(
            self.model.jnt_type[joint]
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            for joint in joints
        ):
            raise ValueError("IK supports hinge and slide joints only")

        indices = self.model.jnt_qposadr[joints]
        limited = self.model.jnt_limited[joints]
        ranges = self.model.jnt_range[joints]
        lower = np.where(limited, ranges[:, 0] + 1e-4, -np.inf)
        upper = np.where(limited, ranges[:, 1] - 1e-4, np.inf)
        if self._ik_data is None:
            self._ik_data = mujoco.MjData(self.model)
        data = self._ik_data
        mujoco.mj_copyData(data, self.model, self.data)
        current = data.qpos[indices].copy()
        position = target_pose.as_translation()
        target_rotation = target_pose.as_rotation()

        def residual(q):
            data.qpos[indices] = q
            mujoco.mj_kinematics(self.model, data)
            rotation = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))
            angle = (target_rotation * rotation.inv()).as_rotvec()
            return np.r_[data.site_xpos[site] - position, angle * 0.18]

        candidates = (current,) if seed is None else (seed, current)
        for candidate in candidates:
            result = least_squares(
                residual,
                np.clip(candidate, lower, upper),
                bounds=(lower, upper),
                max_nfev=180,
                ftol=1e-7,
                xtol=1e-7,
            )
            error = residual(result.x)
            if np.linalg.norm(error[:3]) <= 0.008 and np.linalg.norm(error[3:]) <= 0.06:
                return result.x
        raise ValueError(f"Unreachable IK pose for {self.name}/{frame}: error {error}")
