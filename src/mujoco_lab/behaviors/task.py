"""Physical book-insertion and cube-stacking tasks and their measurements.

The cube geometry and task-5 positions come from OGBench. The released-stack
criterion belongs to this workspace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab.robot import Robot
from mujoco_lab.simulation import Simulator
from mujoco_lab.utils import Transform

DEFAULT_PLACE_HOLD_S = 0.5


def _contact_geoms(data: mujoco.MjData, geom: int) -> set[int]:
    """Return all geoms physically touching one object geom."""
    others = set()
    for contact in data.contact:
        if contact.geom1 == geom:
            others.add(int(contact.geom2))
        elif contact.geom2 == geom:
            others.add(int(contact.geom1))
    return others


class Task[Status](ABC):
    """Track object starts, maximum heights, and stable placement without owning physics."""

    place_hold_s = DEFAULT_PLACE_HOLD_S
    robot: Robot | None

    def __init__(self, simulator: Simulator, body_ids: list[int], geom_ids: list[int]) -> None:
        self.simulator = simulator
        self._body_ids = body_ids
        self._geom_ids = geom_ids
        self._placed = np.zeros(len(body_ids), dtype=bool)
        self.reset()

    def reset(self) -> None:
        """Reset task measurements after the caller resets physics."""
        self.starts = self.simulator.data.xpos[self._body_ids]
        self._max_height = self.starts[:, 2].copy()
        self._stable_since: list[float | None] = [None] * len(self._body_ids)
        self._placed.fill(False)

    def _track_heights(self, heights: np.ndarray) -> None:
        self._max_height = np.maximum(self._max_height, heights)

    def _track_placement(self, index: int, stable: bool | np.bool_, now: float) -> bool:
        if stable:
            if self._stable_since[index] is None:
                self._stable_since[index] = now
        else:
            self._stable_since[index] = None
        since = self._stable_since[index]
        held = since is not None and now - since >= self.place_hold_s
        self._placed[index] |= held
        return held

    def completed_placements(self) -> np.ndarray:
        """Snapshot objects that completed a stable hold since reset, without updating timers."""
        return self._placed.copy()

    def touches_robot(self, index: int = 0, *, contacts: set[int] | None = None) -> bool:
        """Measure object contact with the task's robot, optionally reusing sampled contacts."""
        robot = self.robot
        if robot is None:
            return False
        if contacts is None:
            contacts = _contact_geoms(self.simulator.data, self._geom_ids[index])
        return any(robot.model.geom(geom).name.startswith(robot.prefix) for geom in contacts)

    @abstractmethod
    def status(self) -> Status:
        """Measure task-specific physical conditions and update placement history."""


@dataclass(frozen=True)
class BookStatus:
    """Measured grasp pose and released placement, independent of planner stages.

    The grasp lies at half height, 0.012 m inside the book's front depth face.
    Grip X crosses thickness, grip Y follows height, and grip Z points into depth.
    """

    position_error: float
    rotation_error: float
    is_touch_shelf: bool
    is_touch_robot: bool
    is_released_stable: bool
    grasp_pose: Transform


class BookTask(Task[BookStatus]):
    """Track one physical free book and its shelf site without welding it."""

    position_tolerance_m = 0.015
    rotation_tolerance_rad = 0.12
    linear_speed_tolerance_m_s = 0.01
    angular_speed_tolerance_rad_s = 0.1
    robot: Robot

    def __init__(self, simulator: Simulator) -> None:
        if len(simulator.robots) != 1:
            raise ValueError("Book insertion requires exactly one robot")
        self.robot = next(iter(simulator.robots.values()))
        gripper = self.robot.gripper
        if gripper is None:
            raise ValueError("Book insertion requires a gripper")
        self.gripper = gripper
        model = simulator.model
        self.body = model.body("book").id
        self.geom = model.geom("book/collision").id
        self.target = model.site("book_target").id
        self.support = model.geom("large_shelf/shelf_1").id
        self.joint = model.joint("book/free_joint").id
        if model.jnt_type[self.joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Book must have a free joint")
        self.depth = float(2 * model.geom_size[self.geom, 0])
        super().__init__(simulator, [self.body], [self.geom])

    @property
    def start_center(self) -> np.ndarray:
        return self.starts[0]

    @property
    def max_height(self) -> float:
        return float(self._max_height[0])

    def contact_geoms(self) -> set[int]:
        """Return all geoms physically touching the book collision box."""
        return _contact_geoms(self.simulator.data, self.geom)

    def has_grasp(self) -> bool:
        """Require physical contact on both fingers."""
        return self.gripper.has_contact_on_all_fingers(self.contact_geoms())

    def status(self) -> BookStatus:
        """Report the grasp pose and track stable placement for the configured hold."""
        model, data = self.simulator.model, self.simulator.data
        center = data.xpos[self.body]
        rotation = data.xmat[self.body].reshape(3, 3)
        grasp_pose = Transform(
            rotation=rotation[:, [1, 2, 0]],
            translation=center + rotation[:, 0] * (-self.depth / 2 + 0.012),
        )
        self._track_heights(center[2:3])
        position_error = float(np.linalg.norm(center - data.site_xpos[self.target]))
        book_rotation = Rotation.from_matrix(rotation)
        target_rotation = Rotation.from_matrix(data.site_xmat[self.target].reshape(3, 3))
        relative_rotation = target_rotation * book_rotation.inv()
        # SciPy includes NotImplemented in the result type even for two Rotations.
        rotation_error = float(relative_rotation.magnitude())  # ty: ignore[unresolved-attribute]
        contacts = self.contact_geoms()
        is_touch_shelf = self.support in contacts
        is_touch_robot = self.touches_robot(contacts=contacts)
        adr = int(model.jnt_dofadr[self.joint])
        slow = (
            np.linalg.norm(data.qvel[adr : adr + 3]) < self.linear_speed_tolerance_m_s
            and np.linalg.norm(data.qvel[adr + 3 : adr + 6]) < self.angular_speed_tolerance_rad_s
        )
        stable = (
            position_error < self.position_tolerance_m
            and rotation_error < self.rotation_tolerance_rad
            and is_touch_shelf
            and not is_touch_robot
        )
        held = self._track_placement(0, stable and slow, data.time)
        return BookStatus(
            position_error=position_error,
            rotation_error=rotation_error,
            is_touch_shelf=is_touch_shelf,
            is_touch_robot=is_touch_robot,
            is_released_stable=held,
            grasp_pose=grasp_pose,
        )


@dataclass(frozen=True)
class CubeStatus:
    """Per-cube measurements and overall placement results after a physics step."""

    centers: np.ndarray
    position_errors: np.ndarray
    aligned: np.ndarray
    at_height: np.ndarray
    supported: np.ndarray
    stable_placement: np.ndarray
    is_final_success: bool
    is_released_stable: bool
    is_support_contacts: bool


class CubeStackTask(Task[CubeStatus]):
    """Track cube goals and physical success independently of a controller."""

    xy_tolerance_m = 0.018
    height_tolerance_m = 0.012
    goal_tolerance_m = 0.04
    velocity_norm_tolerance = 0.025  # Combined linear and angular free-joint velocity norm.

    def __init__(self, simulator: Simulator, cubes: int = 2) -> None:
        self.robot = next(iter(simulator.robots.values()), None)
        self.cubes = cubes
        model = simulator.model
        if (
            any(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube{i}/object_0") < 0
                for i in range(cubes)
            )
            or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube{cubes}/object_0") >= 0
        ):
            raise ValueError("cubes must match the loaded cube environment")
        body_ids = [model.body(f"cube{i}/object_0").id for i in range(cubes)]
        geom_ids = [model.geom(f"cube{i}/object_0").id for i in range(cubes)]
        joint_ids = [model.joint(f"cube{i}/object_joint_0").id for i in range(cubes)]
        self._velocity_dofs = model.jnt_dofadr[joint_ids, None] + np.arange(6)
        goal_body_ids = [model.body(f"cube{i}/object_target_0").id for i in range(cubes)]
        self.goals = model.body_pos[goal_body_ids]
        self.cube_size = 2 * model.geom_size[geom_ids[0], 2]
        self._supports = []
        for index, goal in enumerate(self.goals):
            below = [
                other
                for other, candidate in enumerate(self.goals)
                if other != index
                and np.linalg.norm(candidate[:2] - goal[:2]) < self.xy_tolerance_m
                and abs(goal[2] - candidate[2] - self.cube_size) < self.height_tolerance_m
            ]
            self._supports.append("table/box" if not below else f"cube{below[0]}/object_0")
        self._support_geom_ids = [model.geom(name).id for name in self._supports]
        robot_prefixes = tuple(robot.prefix for robot in simulator.robots.values())
        self._robot_geom_ids = {
            geom for geom in range(model.ngeom) if model.geom(geom).name.startswith(robot_prefixes)
        }
        super().__init__(simulator, body_ids, geom_ids)

    @property
    def max_height(self) -> np.ndarray:
        return self._max_height

    def has_grasp(self, cube_index: int, *, robot: Robot | None = None) -> bool:
        """Measure a robot's finger contacts, defaulting to the task's robot."""
        robot = self.robot if robot is None else robot
        if robot is None or robot.gripper is None:
            return False
        contacts = _contact_geoms(robot.data, self._geom_ids[cube_index])
        return robot.gripper.has_contact_on_all_fingers(contacts)

    def status(self) -> CubeStatus:
        """Measure cube placement and track stability for the configured hold."""
        data = self.simulator.data
        centers = data.xpos[self._body_ids]
        position_errors = np.linalg.norm(centers - self.goals, axis=1)
        velocities = np.linalg.norm(data.qvel[self._velocity_dofs], axis=1)
        aligned = np.linalg.norm(centers[:, :2] - self.goals[:, :2], axis=1) < self.xy_tolerance_m
        heights = np.abs(centers[:, 2] - self.goals[:, 2]) < self.height_tolerance_m
        contacts = {geom: set() for geom in self._geom_ids}
        for contact in data.contact:
            if contact.geom1 in contacts:
                contacts[contact.geom1].add(int(contact.geom2))
            if contact.geom2 in contacts:
                contacts[contact.geom2].add(int(contact.geom1))
        support_contacts = np.array(
            [
                support in contacts[geom]
                for geom, support in zip(self._geom_ids, self._support_geom_ids, strict=True)
            ],
            dtype=bool,
        )
        touching_robot = np.array(
            [bool(contacts[geom] & self._robot_geom_ids) for geom in self._geom_ids],
            dtype=bool,
        )
        stable = (
            aligned
            & heights
            & (velocities < self.velocity_norm_tolerance)
            & support_contacts
            & ~touching_robot
        )
        self._track_heights(centers[:, 2])
        held = np.zeros(self.cubes, dtype=bool)
        for index, stable_placement in enumerate(stable):
            held[index] = self._track_placement(index, stable_placement, data.time)
        return CubeStatus(
            centers=centers,
            position_errors=position_errors,
            aligned=aligned,
            at_height=heights,
            supported=support_contacts,
            stable_placement=stable,
            is_final_success=bool(np.all(position_errors <= self.goal_tolerance_m)),
            is_released_stable=bool(np.all(held)),
            is_support_contacts=bool(np.all(support_contacts)),
        )
