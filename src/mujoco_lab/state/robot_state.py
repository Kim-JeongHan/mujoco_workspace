"""Robot-local state and dynamics access over a shared MuJoCo model/data pair."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mujoco_lab.assets.robot.robot import Constraints
from mujoco_lab.state.joint_state import JointState
from mujoco_lab.utils import Transform


class IKError(ValueError):
    """A well-formed target pose could not be reached by inverse kinematics."""


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
    Configured constraints retain their own joint_names ordering.
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
        constraints: Constraints | None = None,
        ik_frame: str = "grasp",
    ) -> None:
        self.model, self.data = model, data
        self.name, self.prefix = name, prefix
        self.joint_names = joint_names
        self.constraints = constraints
        self.ik_frame = ik_frame
        self.root_body_id = model.body(prefix + root_name).id
        bodies = _descendants(model, self.root_body_id)
        self.joint_ids = [model.joint(prefix + name).id for name in joint_names]
        if any(model.jnt_bodyid[j] not in bodies for j in self.joint_ids):
            raise ValueError("Robot joints must belong to its root body subtree")
        unsupported = [
            f"{joint_names[index]} ({mujoco.mjtJoint(model.jnt_type[joint_id]).name})"
            for index, joint_id in enumerate(self.joint_ids)
            if model.jnt_type[joint_id]
            not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        ]
        if unsupported:
            raise ValueError(f"Robot {name!r} supports hinge/slide joints only: {unsupported}")
        self.qpos_indices = [int(model.jnt_qposadr[joint]) for joint in self.joint_ids]
        self.dof_indices = [int(model.jnt_dofadr[joint]) for joint in self.joint_ids]
        self.nq = self.nv = len(self.joint_ids)
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

    def get_joint_limits(self, slots: list[int]) -> np.ndarray:
        """Return configured position limits in the selected robot joint order."""
        constraints = self.constraints
        if constraints is None:
            raise ValueError("Joint limits require configured constraints")
        limits = np.asarray(constraints.position_limit, dtype=float)
        if (
            limits.shape != (len(constraints.joint_names), 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("Configured position limits must be finite with lower < upper")
        try:
            indices = [constraints.joint_names.index(self.joint_names[slot]) for slot in slots]
        except ValueError as error:
            raise ValueError("Position limits must include every selected joint") from error
        return limits[indices].copy()

    def site_id(self, frame: str) -> int:
        """Resolve a local site name belonging to this robot."""
        try:
            return self._sites[frame]
        except KeyError:
            raise ValueError(f"Robot {self.name!r} has no site named {frame!r}") from None

    def get_frame_joint_slots(self, frame: str) -> list[int]:
        """Return robot joint slots on a site's ancestor chain, in joint-name order."""
        site = self.site_id(frame)
        ancestors = set()
        body = int(self.model.site_bodyid[site])
        while body:
            ancestors.add(body)
            body = int(self.model.body_parentid[body])
        slots = [
            slot
            for slot, joint in enumerate(self.joint_ids)
            if self.model.jnt_bodyid[joint] in ancestors
        ]
        if not slots:
            raise ValueError(f"Frame {frame!r} has no movable joints belonging to this robot")
        return slots

    def get_frame_position(self, frame: str) -> np.ndarray:
        """Borrow a site position in world coordinates, in meters."""
        return self.data.site_xpos[self.site_id(frame)]

    def get_jacobian(self, frame: str, dof_slots: Sequence[int] | None = None) -> np.ndarray:
        """Get the robot Jacobian or selected robot-local DOF columns.

        The full result borrows a persistent buffer; a selected result is a copy.
        Linear rows precede angular rows.
        """

        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._full_jacobian[:3],
            self._full_jacobian[3:],
            self.site_id(frame),
        )
        self._jacobian[:] = self._full_jacobian[:, self.dof_indices]
        return self._jacobian if dof_slots is None else self._jacobian[:, dof_slots]

    def get_mass_matrix(self, dof_slots: Sequence[int] | None = None) -> np.ndarray:
        """Get the robot mass block or selected robot-local DOF block.

        The full result borrows a persistent buffer; a selected result is a copy.
        """
        mujoco.mj_fullM(self.model, self.data, self._full_mass)
        self._mass[:] = self._full_mass[np.ix_(self.dof_indices, self.dof_indices)]
        return self._mass if dof_slots is None else self._mass[np.ix_(dof_slots, dof_slots)]

    def solve_ik(
        self,
        target_pose: Transform,
        reference_q: ArrayLike,
        *,
        retry_current: bool = True,
    ) -> np.ndarray:
        """Solve a world-space site pose without modifying shared simulation data.

        Only this robot's hinge/slide joints on the site's ancestor chain are
        optimized; descendant gripper joints and other robots remain fixed.
        The configured ik_frame selects the site to match. Returned positions
        and the required reference_q follow the ancestor joints' joint_names order.
        YAML position limits bound every IK joint with a 1e-4 margin.

        The reference posture is tried first. Unless retry_current is False, a
        distinct current posture is tried after an unreachable reference. Callers
        comparing multiple reference postures can disable this internal retry.
        Rotation-vector errors have weight 0.18 in the least-squares residual,
        with acceptance at 8 mm position error and 0.06 weighted rotation error
        (1/3 rad). Raise ValueError if no attempt meets these tolerances.
        The result is not collision-checked. An unreachable target raises
        IKError, a ValueError subclass; configuration errors remain distinct.
        """
        frame = self.ik_frame
        site = self.site_id(frame)
        slots = self.get_frame_joint_slots(frame)
        joints = [self.joint_ids[slot] for slot in slots]
        indices = self.model.jnt_qposadr[joints]
        limits = self.get_joint_limits(slots)
        lower = limits[:, 0] + 1e-4
        upper = limits[:, 1] - 1e-4
        if self._ik_data is None:
            self._ik_data = mujoco.MjData(self.model)
        data = self._ik_data
        mujoco.mj_copyData(data, self.model, self.data)
        current = data.qpos[indices].copy()
        reference = np.asarray(reference_q, dtype=float)
        if reference.shape != current.shape or not np.isfinite(reference).all():
            raise ValueError("reference_q must contain one finite position per IK joint")
        position = target_pose.as_translation()
        target_rotation = target_pose.as_rotation()

        def residual(q):
            data.qpos[indices] = q
            mujoco.mj_kinematics(self.model, data)
            rotation = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))
            relative_rotation = target_rotation * rotation.inv()
            # SciPy includes NotImplemented in the result type even for two Rotations.
            angle = relative_rotation.as_rotvec()  # ty: ignore[unresolved-attribute]
            return np.r_[data.site_xpos[site] - position, angle * 0.18]

        initial = np.clip(reference, lower, upper)
        candidates = [initial]
        if retry_current:
            current = np.clip(current, lower, upper)
            if not np.array_equal(initial, current):
                candidates.append(current)
        for candidate in candidates:
            result = least_squares(
                residual,
                candidate,
                bounds=(lower, upper),
                max_nfev=180,
                ftol=1e-7,
                xtol=1e-7,
            )
            error = residual(result.x)
            if np.linalg.norm(error[:3]) <= 0.008 and np.linalg.norm(error[3:]) <= 0.06:
                return result.x
        raise IKError(f"Unreachable IK pose for {self.name}/{frame}: error {error}")

    def get_constraint(self, index: int) -> tuple[str, list[float], list[float], list[float]]:
        """Return a configured constraint by index, in the order they were added."""
        if self.constraints is None:
            raise ValueError("Robot state has no configured constraints")
        return self.constraints.get_info(index)
