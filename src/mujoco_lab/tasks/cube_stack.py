"""Physical cube goals and released-stack measurements for OGBench layouts.

The cube geometry and task-5 positions come from OGBench. The released-stack
criterion belongs to this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_lab.simulation import Simulator


@dataclass(frozen=True)
class StackStatus:
    """Task measurements after a physics step."""

    goal_distances: np.ndarray
    ogbench_success: bool
    released_stable_stack: bool
    support_contacts: bool
    cube_heights: np.ndarray


@dataclass(frozen=True)
class CubeMeasurements:
    """Read-only per-cube measurements used by task success and evaluation."""

    centers: np.ndarray
    goal_distances: np.ndarray
    velocities: np.ndarray
    aligned: np.ndarray
    at_height: np.ndarray
    supported: np.ndarray
    touching_robot: np.ndarray
    stable_placement: np.ndarray


class CubeStackTask:
    """Track cube goals and physical success independently of a controller."""

    def __init__(self, simulator: Simulator, cubes: int = 2):
        self.simulator = simulator
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
        self.starts = np.array([model.body(f"cube{i}/object_0").pos for i in range(cubes)])
        self.goals = np.array([model.body(f"cube{i}/object_target_0").pos for i in range(cubes)])
        self.cube_size = 2 * model.geom("cube0/object_0").size[2]
        self._supports = []
        for index, goal in enumerate(self.goals):
            below = [
                other
                for other, candidate in enumerate(self.goals)
                if other != index
                and np.linalg.norm(candidate[:2] - goal[:2]) < 0.018
                and abs(goal[2] - candidate[2] - self.cube_size) < 0.012
            ]
            self._supports.append("table/box" if not below else f"cube{below[0]}/object_0")
        self.max_lift = self.starts[:, 2].copy()
        self._stable_since = None

    def measurements(self) -> CubeMeasurements:
        """Measure cube placement without changing task history or simulation state."""
        data, model = self.simulator.data, self.simulator.model
        centers = np.array([data.body(f"cube{i}/object_0").xpos for i in range(self.cubes)])
        distances = np.linalg.norm(centers - self.goals, axis=1)
        velocities = np.array(
            [
                np.linalg.norm(
                    data.qvel[int(model.joint(f"cube{i}/object_joint_0").dofadr[0]) :][:6]
                )
                for i in range(self.cubes)
            ]
        )
        aligned = np.linalg.norm(centers[:, :2] - self.goals[:, :2], axis=1) < 0.018
        heights = np.abs(centers[:, 2] - self.goals[:, 2]) < 0.012
        contacts = {
            frozenset((model.geom(contact.geom1).name, model.geom(contact.geom2).name))
            for contact in data.contact
        }
        support_contacts = np.array(
            [
                frozenset((self._supports[i], f"cube{i}/object_0")) in contacts
                for i in range(self.cubes)
            ],
            dtype=bool,
        )
        robot_prefixes = tuple(robot.prefix for robot in self.simulator.robots.values())
        touching_robot = np.array(
            [
                any(
                    f"cube{i}/object_0" in pair
                    and any(name.startswith(robot_prefixes) for name in pair)
                    for pair in contacts
                )
                for i in range(self.cubes)
            ],
            dtype=bool,
        )
        stable = aligned & heights & (velocities < 0.025) & support_contacts & ~touching_robot
        return CubeMeasurements(
            centers,
            distances,
            velocities,
            aligned,
            heights,
            support_contacts,
            touching_robot,
            stable,
        )

    def status(self) -> StackStatus:
        """Sample physical state; call regularly to track the 0.5-second stable hold."""
        data = self.simulator.data
        sample = self.measurements()
        self.max_lift = np.maximum(self.max_lift, sample.centers[:, 2])
        stable = bool(np.all(sample.stable_placement))
        if stable:
            if self._stable_since is None:
                self._stable_since = data.time
        else:
            self._stable_since = None
        return StackStatus(
            sample.goal_distances,
            bool(np.all(sample.goal_distances <= 0.04)),
            self._stable_since is not None and data.time - self._stable_since >= 0.5,
            bool(np.all(sample.supported)),
            sample.centers[:, 2].copy(),
        )

    def reset(self) -> None:
        """Reset physics and measurements; expert.reset() then replans its script."""
        self.simulator.reset()
        self.max_lift[:] = self.starts[:, 2]
        self._stable_since = None
