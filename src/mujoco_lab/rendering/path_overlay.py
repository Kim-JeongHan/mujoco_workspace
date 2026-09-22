"""Planned and measured end-effector paths in a MuJoCo scene."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import mujoco
import numpy as np

from mujoco_lab.rendering.annotations import add_marker
from mujoco_lab.robot import Robot

PLANNED_RGBA = (0.10, 0.72, 1.00, 0.95)
TRAIL_RGBA = (1.00, 0.48, 0.10, 0.95)
START_RGBA = (0.10, 0.72, 1.00, 1.00)
GOAL_RGBA = (0.35, 1.00, 0.25, 1.00)
CURRENT_RGBA = (1.00, 0.28, 0.10, 1.00)
_ZERO = np.zeros(3)
_ROTATION = np.eye(3).ravel()
_PLANNED_COLOR = np.asarray(PLANNED_RGBA, dtype=np.float32)
_TRAIL_COLOR = np.asarray(TRAIL_RGBA, dtype=np.float32)


def _line(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray, rgba, width: float) -> None:
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-8:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        _ZERO,
        _ZERO,
        _ROTATION,
        rgba,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
    scene.ngeom += 1


class PathOverlay:
    """Draw a joint-space plan and a bounded measured endpoint trail.

    ``joint_names`` and ``waypoints`` share the planner's joint order. The
    selected frame is a robot-local site or body name; by default a grasp
    site is preferred, followed by ee_site and the UR tool0 body. Planned
    FK starts from a copy of the current simulation state, preserving all
    joints outside the plan, and never writes to live data. On scenes with a
    horizontal world floor, an XY projection keeps paths visible when the
    robot's own meshes hide the true 3D endpoint path.
    """

    def __init__(
        self,
        robot: Robot,
        joint_names: Sequence[str],
        waypoints: Sequence[Sequence[float]] | np.ndarray,
        frame: str | None = None,
    ) -> None:
        self.model = robot.model
        self._frame_kind, self._frame_id = self._resolve_frame(robot, frame)
        names = tuple(joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be nonempty and unique")
        if any(name not in robot.state.joint_names for name in names):
            raise ValueError("joint_names must belong to the selected robot")
        joint_ids = [self.model.joint(robot.prefix + name).id for name in names]
        if any(
            int(self.model.jnt_type[j])
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            for j in joint_ids
        ):
            raise ValueError("PathOverlay supports scalar hinge and slide joints")
        self.qpos_indices = np.asarray(self.model.jnt_qposadr[joint_ids], dtype=int)
        q = np.asarray(waypoints, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(names) or len(q) == 0 or not np.isfinite(q).all():
            raise ValueError("waypoints must be a nonempty finite array in joint_names order")

        scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(scratch, self.model, robot.data)
        samples = self._joint_samples(q)
        planned = np.empty((len(samples), 3))
        for i, sample in enumerate(samples):
            scratch.qpos[self.qpos_indices] = sample
            mujoco.mj_kinematics(self.model, scratch)
            planned[i] = self._position(scratch)
        self.planned_xyz = planned
        self.floor_z = self._floor_height()
        self._trail: deque[np.ndarray] = deque(maxlen=180)
        self.current_xyz: np.ndarray | None = None

    def _floor_height(self) -> float | None:
        """Find a horizontal world plane for the optional XY path projection."""
        for geom in range(self.model.ngeom):
            if (
                self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE
                and self.model.geom_bodyid[geom] == 0
                and np.linalg.norm(self.model.geom_quat[geom, 1:3]) < 1e-6
            ):
                return float(self.model.geom_pos[geom, 2])
        return None

    @staticmethod
    def _project(points: np.ndarray, height: float) -> np.ndarray:
        projected = points.copy()
        projected[..., 2] = height
        return projected

    @staticmethod
    def _resolve_frame(robot: Robot, frame: str | None) -> tuple[str, int]:
        model = robot.model
        candidates = (frame,) if frame is not None else ("grasp", "ee_site", "tool0")
        for name in candidates:
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.prefix + name)
            if site >= 0:
                body = int(model.site_bodyid[site])
                kind, frame_id = "site", site
            else:
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.prefix + name)
                kind, frame_id = "body", body
            if body < 0:
                continue
            owner = body
            while owner > robot.state.root_body_id:
                owner = int(model.body_parentid[owner])
            if owner == robot.state.root_body_id:
                return kind, frame_id
        raise ValueError(f"Robot {robot.name!r} has no endpoint frame {frame!r}")

    @staticmethod
    def _joint_samples(q: np.ndarray) -> np.ndarray:
        samples = [q[0]]
        for start, end in zip(q[:-1], q[1:]):
            subdivisions = min(8, max(1, int(np.ceil(np.linalg.norm(end - start) / 0.15))))
            samples.extend(
                start + (end - start) * (i / subdivisions) for i in range(1, subdivisions + 1)
            )
        result = np.asarray(samples)
        if len(result) > 200:
            result = result[np.linspace(0, len(result) - 1, 200, dtype=int)]
        return result

    def _position(self, data: mujoco.MjData) -> np.ndarray:
        if self._frame_kind == "site":
            return data.site_xpos[self._frame_id].copy()
        return data.xpos[self._frame_id].copy()

    @property
    def trail_xyz(self) -> np.ndarray:
        """Return the retained measured world-space endpoint positions."""
        return np.asarray(self._trail).reshape(-1, 3).copy()

    def record(self, data: mujoco.MjData) -> None:
        """Remember the current measured endpoint without changing simulation state."""
        point = self._position(data)
        self.current_xyz = point
        if not self._trail or np.linalg.norm(point - self._trail[-1]) >= 0.003:
            self._trail.append(point)

    def __call__(self, scene: mujoco.MjvScene, data: mujoco.MjData) -> None:
        """Append path geoms to a scene; the caller owns scene clearing."""
        self.record(data)
        for start, end in zip(self.planned_xyz[:-1], self.planned_xyz[1:]):
            _line(scene, start, end, _PLANNED_COLOR, 0.004)
        trail = self.trail_xyz
        for start, end in zip(trail[:-1], trail[1:]):
            _line(scene, start, end, _TRAIL_COLOR, 0.006)
        if len(trail) and self.current_xyz is not None:
            _line(scene, trail[-1], self.current_xyz, _TRAIL_COLOR, 0.006)

        if self.floor_z is not None:
            # Layer the measured trace just above the wider planned trace.
            # Both preserve the same XY coordinates as their world paths.
            plan_floor = self._project(self.planned_xyz, self.floor_z + 0.016)
            trail_floor = self._project(trail, self.floor_z + 0.040)
            for start, end in zip(plan_floor[:-1], plan_floor[1:]):
                _line(scene, start, end, _PLANNED_COLOR, 0.008)
            for start, end in zip(trail_floor[:-1], trail_floor[1:]):
                _line(scene, start, end, _TRAIL_COLOR, 0.004)
            if len(trail_floor) and self.current_xyz is not None:
                current_floor = self._project(self.current_xyz, self.floor_z + 0.040)
                _line(scene, trail_floor[-1], current_floor, _TRAIL_COLOR, 0.004)
            if scene.ngeom < scene.maxgeom:
                add_marker(scene, plan_floor[0], START_RGBA, 0.012)
                scene.geoms[scene.ngeom - 1].label = "XY floor paths"
            if scene.ngeom < scene.maxgeom:
                add_marker(scene, plan_floor[-1], GOAL_RGBA, 0.013)
            if self.current_xyz is not None and scene.ngeom < scene.maxgeom:
                add_marker(scene, current_floor, CURRENT_RGBA, 0.012)

        for position, rgba, label, size in (
            (self.planned_xyz[0], START_RGBA, "Plan start", 0.019),
            (self.planned_xyz[-1], GOAL_RGBA, "Plan goal", 0.023),
            (self.current_xyz, CURRENT_RGBA, "Actual EEF", 0.017),
        ):
            if position is None or scene.ngeom >= scene.maxgeom:
                break
            add_marker(scene, position, rgba, size)
            # Text at nearby endpoints overlaps in the viewer. Keep every
            # marker visible, and suppress only redundant text.
            if label == "Actual EEF" or np.linalg.norm(position - self.current_xyz) > 0.12:
                scene.geoms[scene.ngeom - 1].label = label
