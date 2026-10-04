"""Measure free-book contacts, grasps, and released shelf placement."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab.assets.robot.robot import ControllerConfig
from mujoco_lab.control import Controller, create_controller
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.simulation import Simulator
from mujoco_lab.utils import Transform


def create_book_controller(robot, config: ControllerConfig) -> Controller:
    """Create book control with payload PD gains."""
    controller = create_controller(robot, config)
    assert controller is not None
    if isinstance(controller, JointSpacePD):
        controller.kp *= 4
        controller.kd *= 2
        controller.kp[4:] *= 6
        controller.kd[4:] *= np.sqrt(6)
    return controller


def center_grasp_pose(
    center: np.ndarray, rotation: np.ndarray, depth: float, *, inset: float = 0.012
) -> Transform:
    """Grasp at half height from the front depth edge, closing across thickness.

    Grip X crosses book thickness, grip Y follows book height, and grip Z
    points into its depth. The site lies ``inset`` meters inside the front
    face. This clears the palm while keeping the grip at the book's Z center.
    """
    return Transform(
        rotation=rotation[:, [1, 2, 0]],
        translation=center + rotation[:, 0] * (-depth / 2 + inset),
    )


@dataclass(frozen=True)
class BookStatus:
    """Measured released placement, independent of planner stage completion."""

    position_error: float
    rotation_error: float
    supported: bool
    touching_robot: bool
    released_stable: bool


class BookTask:
    """Track one physical free book and its shelf site without welding it."""

    def __init__(self, simulator: Simulator) -> None:
        if len(simulator.robots) != 1:
            raise ValueError("Book insertion requires exactly one robot")
        self.simulator = simulator
        self.robot = next(iter(simulator.robots.values()))
        model = simulator.model
        self.body = model.body("book").id
        self.geom = model.geom("book/collision").id
        self.target = model.site("book_target").id
        self.support = model.geom("large_shelf/shelf_1").id
        self.joint = model.joint("book/free_joint").id
        if model.jnt_type[self.joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Book must have a free joint")
        self.depth = float(2 * model.geom_size[self.geom, 0])
        self.reset()

    def reset(self) -> None:
        """Reset task measurements after the caller resets physics."""
        self.start_center = self.simulator.data.xpos[self.body].copy()
        self.max_height = float(self.start_center[2])
        self._stable_since = None

    def contact_geoms(self) -> set[int]:
        """Return all geoms physically touching the book collision box."""
        others = set()
        for contact in self.simulator.data.contact:
            if contact.geom1 == self.geom:
                others.add(int(contact.geom2))
            elif contact.geom2 == self.geom:
                others.add(int(contact.geom1))
        return others

    def has_grasp(self) -> bool:
        """Require physical contact on both finger pads, not a closing command."""
        model = self.simulator.model
        contacts = self.contact_geoms()
        if self.robot.robot_type == "forte":
            return all(
                model.geom(f"{self.robot.prefix}gripper_{side}_pad").id in contacts
                for side in ("left", "right")
            )
        bodies = {int(model.geom_bodyid[g]) for g in contacts}
        return all(
            model.body(f"{self.robot.prefix}panda_{side}finger").id in bodies
            for side in ("left", "right")
        )

    def status(self) -> BookStatus:
        """Require pose, support, release, low velocity, and 0.5 s dwell."""
        model, data = self.simulator.model, self.simulator.data
        self.max_height = max(self.max_height, float(data.xpos[self.body, 2]))
        position_error = float(np.linalg.norm(data.xpos[self.body] - data.site_xpos[self.target]))
        book_rotation = Rotation.from_matrix(data.xmat[self.body].reshape(3, 3))
        target_rotation = Rotation.from_matrix(data.site_xmat[self.target].reshape(3, 3))
        rotation_error = float((target_rotation * book_rotation.inv()).magnitude())
        contacts = self.contact_geoms()
        supported = self.support in contacts
        touching = any(model.geom(g).name.startswith(self.robot.prefix) for g in contacts)
        adr = int(model.jnt_dofadr[self.joint])
        slow = (
            np.linalg.norm(data.qvel[adr : adr + 3]) < 0.01
            and np.linalg.norm(data.qvel[adr + 3 : adr + 6]) < 0.1
        )
        stable = position_error < 0.015 and rotation_error < 0.12 and supported and not touching
        if stable and slow:
            if self._stable_since is None:
                self._stable_since = data.time
        else:
            self._stable_since = None
        return BookStatus(
            position_error,
            rotation_error,
            supported,
            touching,
            self._stable_since is not None and data.time - self._stable_since >= 0.5,
        )
