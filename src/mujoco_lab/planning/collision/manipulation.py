"""Stage-specific collision checks for physical cube stacking."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from .mujoco import MuJoCoCollisionChecker


class CubeStackCollisionChecker(MuJoCoCollisionChecker):
    """Check an arm path with narrow grasp and support contact allowances.

    Construct carry stages only after observing a physical two-finger grasp.
    The active cube follows that measured grasp-to-cube pose only in private
    MuJoCo data. Call ``refresh`` at a new stage boundary to capture the current
    scene and, when carrying, the current relative pose.
    """

    _CARRY_STAGES = frozenset(("lift", "above_place", "place"))
    _BOUND_EPS = 1e-5  # MuJoCo's soft joint limits permit tiny measured overshoots.
    _SELF_CONTACT_EPS = 1e-5  # Ignore mesh tessellation contact at numerical scale.
    _GRASP_STAGES = frozenset(("pick", "close", "lift", "above_place", "place", "release"))
    _STAGES = frozenset(
        ("above_pick", "pick", "close", "lift", "above_place", "place", "release", "retract")
    )

    def __init__(
        self,
        robot,
        stage_name: str,
        *,
        bounds: Sequence[Sequence[float]] | None = None,
        edge_resolution: float = 0.05,
        support_geom: str | int | None = None,
        grasp_geoms: Sequence[str | int] | None = None,
        grasp_penetration: float = 0.012,
        support_penetration: float = 0.004,
        support_xy_tolerance: float = 0.01,
    ) -> None:
        if not np.isfinite(grasp_penetration) or grasp_penetration < 0:
            raise ValueError("grasp_penetration must be finite and nonnegative")
        if not np.isfinite(support_penetration) or support_penetration < 0:
            raise ValueError("support_penetration must be finite and nonnegative")
        if not np.isfinite(support_xy_tolerance) or support_xy_tolerance <= 0:
            raise ValueError("support_xy_tolerance must be finite and positive")
        self._grasp_penetration = float(grasp_penetration)
        self._support_penetration = float(support_penetration)
        self._support_xy_tolerance = float(support_xy_tolerance)
        self._support_name = support_geom
        self._grasp_names = grasp_geoms
        self._configured = False

        if bounds is None:
            model, state = robot.model, robot.state
            site = state.site_id("grasp")
            ancestors = set()
            body = int(model.site_bodyid[site])
            while body:
                ancestors.add(body)
                body = int(model.body_parentid[body])
            selected = [j for j in state.joint_ids if model.jnt_bodyid[j] in ancestors]
            bounds = []
            for joint in selected:
                if model.jnt_limited[joint]:
                    bounds.append(tuple(model.jnt_range[joint]))
                else:
                    current = robot.data.qpos[model.jnt_qposadr[joint]]
                    bounds.append((float(current - np.pi), float(current + np.pi)))

        super().__init__(robot, frame="grasp", bounds=bounds, edge_resolution=edge_resolution)
        self._site_id = robot.state.site_id("grasp")
        self._set_stage(stage_name)
        self._configured = True
        self.refresh()

    def _set_stage(self, stage_name: str) -> None:
        cube, separator, phase = stage_name.partition(":")
        if (
            not separator
            or not cube.startswith("cube")
            or not cube[4:].isdigit()
            or phase not in self._STAGES
        ):
            raise ValueError(f"Invalid cube stack stage: {stage_name!r}")
        model = self.model
        cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{cube}/object_0")
        if cube_body < 0:
            raise ValueError(f"Missing cube body for {stage_name!r}")
        joint_start = int(model.body_jntadr[cube_body])
        if (
            model.body_jntnum[cube_body] != 1
            or model.jnt_type[joint_start] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError("Active cube must have one free joint")
        self.stage_name = stage_name
        self._phase = phase
        self._cube_body = cube_body
        self._cube_qpos_adr = int(model.jnt_qposadr[joint_start])
        cube_owned = np.zeros(model.nbody, dtype=bool)
        cube_owned[cube_body] = True
        for body in range(cube_body + 1, model.nbody):
            cube_owned[body] = cube_owned[model.body_parentid[body]]
        self._cube_geoms = cube_owned[model.geom_bodyid]
        self._grasp_geom_ids = self._resolve_grasp_geoms()
        support = self._support_name
        if support is None and phase == "lift":
            support = "table/box"
        self._support_geom_id = self._geom_id(support) if support is not None else None
        if self._support_geom_id is not None and (
            self._owned_geoms[self._support_geom_id] or self._cube_geoms[self._support_geom_id]
        ):
            raise ValueError("support_geom must belong to the surroundings")
        self._target_body = None
        if phase == "place" and self._support_geom_id is not None:
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{cube}/object_target_0")
            if target < 0:
                raise ValueError(f"Missing cube target body for {stage_name!r}")
            self._target_body = target

    def _geom_id(self, value: str | int) -> int:
        if isinstance(value, str):
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, value)
        else:
            geom = int(value)
        if geom < 0 or geom >= self.model.ngeom:
            raise ValueError(f"Unknown collision geom: {value!r}")
        return geom

    def _resolve_grasp_geoms(self) -> frozenset[int]:
        if self._grasp_names is not None:
            ids = frozenset(self._geom_id(value) for value in self._grasp_names)
        elif self.robot.robot_type == "panda":
            finger_bodies = {
                int(self.model.jnt_bodyid[joint])
                for joint in self.robot.state.joint_ids
                if self.model.joint(joint).name.endswith(
                    ("panda_finger_joint1", "panda_finger_joint2")
                )
            }
            ids = frozenset(
                geom
                for geom in range(self.model.ngeom)
                if self.model.geom_bodyid[geom] in finger_bodies
                and self.model.geom_contype[geom]
                and self.model.geom_conaffinity[geom]
            )
        elif self.robot.robot_type == "forte":
            ids = frozenset(
                self._geom_id(f"{self.robot.name}/gripper_{side}_pad") for side in ("left", "right")
            )
        else:
            raise ValueError("Cube stacking requires panda or forte, or explicit grasp_geoms")
        if len(ids) != 2 or not all(self._owned_geoms[geom] for geom in ids):
            raise ValueError("grasp_geoms must identify two robot finger or pad geoms")
        return ids

    def refresh(self, stage_name: str | None = None) -> None:
        """Capture live positions and the current grasp-to-cube pose."""
        super().refresh()
        if not self._configured:
            return
        if stage_name is not None:
            self._set_stage(stage_name)
        if self._phase not in self._CARRY_STAGES:
            return
        mujoco.mj_fwdPosition(self.model, self._snapshot)
        if self._phase == "lift":
            self._support_xy = self._snapshot.xpos[self._cube_body, :2].copy()
        elif self._phase == "place" and self._target_body is not None:
            self._support_xy = self._snapshot.xpos[self._target_body, :2].copy()
        site_pos = self._snapshot.site_xpos[self._site_id]
        site_quat = np.empty(4)
        mujoco.mju_mat2Quat(site_quat, self._snapshot.site_xmat[self._site_id])
        inv_pos, inv_quat = np.empty(3), np.empty(4)
        mujoco.mju_negPose(inv_pos, inv_quat, site_pos, site_quat)
        self._relative_pos, self._relative_quat = np.empty(3), np.empty(4)
        mujoco.mju_mulPose(
            self._relative_pos,
            self._relative_quat,
            inv_pos,
            inv_quat,
            self._snapshot.xpos[self._cube_body],
            self._snapshot.xquat[self._cube_body],
        )

    def is_collision_free(self, state: np.ndarray) -> bool:
        """Check the arm and, during carry, a virtual attached cube."""
        q = self._state(state)
        if np.any(q < self.bounds[:, 0] - self._BOUND_EPS) or np.any(
            q > self.bounds[:, 1] + self._BOUND_EPS
        ):
            return False
        mujoco.mj_copyData(self._scratch, self.model, self._snapshot)
        self._scratch.qpos[self.qpos_indices] = q
        mujoco.mj_fwdPosition(self.model, self._scratch)
        if self._phase in self._CARRY_STAGES:
            site_pos = self._scratch.site_xpos[self._site_id]
            site_quat = np.empty(4)
            mujoco.mju_mat2Quat(site_quat, self._scratch.site_xmat[self._site_id])
            cube_pos, cube_quat = np.empty(3), np.empty(4)
            mujoco.mju_mulPose(
                cube_pos,
                cube_quat,
                site_pos,
                site_quat,
                self._relative_pos,
                self._relative_quat,
            )
            adr = self._cube_qpos_adr
            self._scratch.qpos[adr : adr + 3] = cube_pos
            self._scratch.qpos[adr + 3 : adr + 7] = cube_quat
            mujoco.mj_fwdPosition(self.model, self._scratch)
        for contact in self._scratch.contact[: self._scratch.ncon]:
            if contact.dist > 0:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            robot_contact = self._owned_geoms[first] or self._owned_geoms[second]
            cube_contact = self._cube_geoms[first] or self._cube_geoms[second]
            if not robot_contact and not (self._phase in self._CARRY_STAGES and cube_contact):
                continue
            pair = frozenset((first, second))
            if (
                self._owned_geoms[first]
                and self._owned_geoms[second]
                and contact.dist >= -self._SELF_CONTACT_EPS
            ):
                continue
            if (
                self._phase in self._GRASP_STAGES
                and cube_contact
                and any(geom in pair for geom in self._grasp_geom_ids)
                and contact.dist >= -self._grasp_penetration
            ):
                continue
            if (
                self._phase in ("lift", "place")
                and self._support_geom_id is not None
                and cube_contact
                and self._support_geom_id in pair
                and contact.dist >= -self._support_penetration
                and np.linalg.norm(self._scratch.xpos[self._cube_body, :2] - self._support_xy)
                <= self._support_xy_tolerance
            ):
                continue
            return False
        return True
