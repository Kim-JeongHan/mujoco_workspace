"""MuJoCo collision checking for a robot's selected joint coordinates."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_lab.robot import Robot

from .collision_checker import CollisionChecker


class MuJoCoCollisionChecker(CollisionChecker):
    """Check a robot against itself and its frozen MuJoCo surroundings.

    Select hinge/slide joints by local names or by the ancestors of a robot site.
    Unselected joints, other robots, and objects retain their snapshotted poses.
    Call ``refresh()`` after changing the live scene and before planning again.
    Native MuJoCo contacts at or beyond touching are rejected; the model's
    collision filters and exclusions remain in force. Edge checks sample a
    straight line in joint coordinates, so clearance between samples is not
    guaranteed. Continuous angles are not wrapped. This class does not advance
    or modify the live simulation.
    """

    def __init__(
        self,
        robot: Robot,
        *,
        frame: str | None = None,
        joint_names: Sequence[str] | None = None,
        bounds: Sequence[Sequence[float]] | None = None,
        edge_resolution: float = 0.1,
    ) -> None:
        if (frame is None) == (joint_names is None):
            raise ValueError("Choose exactly one of frame or joint_names")
        self.robot = robot
        self.model = robot.model
        self._live_data = robot.data
        state = robot.state

        if frame is not None:
            site = state.site_id(frame)
            ancestors = set()
            body = int(self.model.site_bodyid[site])
            while body:
                ancestors.add(body)
                body = int(self.model.body_parentid[body])
            selected = [j for j in state.joint_ids if self.model.jnt_bodyid[j] in ancestors]
        else:
            names = tuple(joint_names)
            if not names or len(names) != len(set(names)):
                raise ValueError("joint_names must be nonempty and unique")
            unknown = set(names) - set(state.joint_names)
            if unknown:
                raise ValueError(f"Joints do not belong to robot {robot.name!r}: {sorted(unknown)}")
            selected = [state.joint_ids[state.joint_names.index(name)] for name in names]

        if not selected:
            raise ValueError("No robot joints selected")
        if any(
            self.model.jnt_type[j]
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            for j in selected
        ):
            raise ValueError("Planning coordinates must be hinge or slide joints")
        self.joint_ids = tuple(selected)
        self.joint_names = tuple(
            self.model.joint(j).name.removeprefix(robot.prefix) for j in selected
        )
        self.qpos_indices = np.asarray(self.model.jnt_qposadr[selected], dtype=int)

        if not np.isfinite(edge_resolution) or edge_resolution <= 0:
            raise ValueError("edge_resolution must be finite and positive")
        self.edge_resolution = float(edge_resolution)

        limited = np.asarray(self.model.jnt_limited[selected], dtype=bool)
        hardware = np.asarray(self.model.jnt_range[selected], dtype=float)
        if bounds is None:
            if not limited.all():
                raise ValueError("Continuous joints require explicit finite bounds")
            requested = hardware.copy()
        else:
            requested = np.asarray(bounds, dtype=float)
            if requested.shape != (len(selected), 2):
                raise ValueError(f"bounds must have shape ({len(selected)}, 2)")
        if not np.isfinite(requested).all() or np.any(requested[:, 0] >= requested[:, 1]):
            raise ValueError("Each planning bound must be finite with lower < upper")
        if np.any(
            limited
            & (
                (requested[:, 0] < hardware[:, 0] - 1e-9)
                | (requested[:, 1] > hardware[:, 1] + 1e-9)
            )
        ):
            raise ValueError("Planning bounds cannot exceed limited joint ranges")
        self.bounds = requested.copy()

        owned = np.zeros(self.model.nbody, dtype=bool)
        owned[state.root_body_id] = True
        for body in range(state.root_body_id + 1, self.model.nbody):
            owned[body] = owned[self.model.body_parentid[body]]
        self._owned_geoms = owned[self.model.geom_bodyid]

        self._snapshot = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        self.refresh()

    def refresh(self) -> None:
        """Capture the current scene, including other robots and free objects."""
        mujoco.mj_copyData(self._snapshot, self.model, self._live_data)

    def _state(self, state: ArrayLike) -> np.ndarray:
        q = np.asarray(state, dtype=float)
        if q.shape != (len(self.joint_ids),) or not np.isfinite(q).all():
            raise ValueError(f"State must contain {len(self.joint_ids)} finite joint values")
        return q

    def is_collision_free(self, state: np.ndarray) -> bool:
        """Check selected joint values in the frozen scene."""
        q = self._state(state)
        if np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            return False
        mujoco.mj_copyData(self._scratch, self.model, self._snapshot)
        self._scratch.qpos[self.qpos_indices] = q
        mujoco.mj_fwdPosition(self.model, self._scratch)
        for contact in self._scratch.contact[: self._scratch.ncon]:
            if contact.dist > 0:
                continue
            if self._owned_geoms[contact.geom1] or self._owned_geoms[contact.geom2]:
                return False
        return True

    def is_path_collision_free(
        self,
        from_state: np.ndarray,
        to_state: np.ndarray,
        resolution: float | None = None,
    ) -> bool:
        """Sample a straight joint-space edge at the chosen finite resolution."""
        start, goal = self._state(from_state), self._state(to_state)
        if resolution is None:
            resolution = self.edge_resolution
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        steps = max(1, int(np.ceil(np.linalg.norm(goal - start) / resolution)))
        return all(
            self.is_collision_free(start + (goal - start) * (i / steps)) for i in range(steps + 1)
        )
