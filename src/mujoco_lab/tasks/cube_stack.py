"""Physical robot manipulation of the OGBench cube stacking layouts.

The cube geometry and task-5 positions come from OGBench. Robot control and
the released-stack criterion belong to this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab.control import create_controller
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.simulation import Simulator
from mujoco_lab.utils import Transform


@dataclass(frozen=True)
class StackStatus:
    """Task measurements after a physics step."""

    goal_distances: np.ndarray
    ogbench_success: bool
    released_stable_stack: bool
    support_contacts: bool
    cube_heights: np.ndarray


class CubeStackTask:
    """Track task-5 targets and script physical pick, lift, place, release."""

    def __init__(
        self,
        simulator: Simulator,
        cubes: int = 2,
        *,
        method: str = "heuristic",
        planner: str = "rrt_connect",
        seed: int = 7,
        planning_budget: int = 500,
    ):
        if cubes not in (2, 3, 4):
            raise ValueError("cubes must be 2, 3, or 4")
        if method not in ("heuristic", "sampling"):
            raise ValueError("method must be heuristic or sampling")
        if planner not in ("rrt_connect", "rrt", "prm"):
            raise ValueError("planner must be rrt_connect, rrt, or prm")
        if planning_budget <= 0:
            raise ValueError("planning_budget must be positive")
        self.simulator = simulator
        self.cubes = cubes
        self.method = method
        self.planner = planner
        self.seed = seed
        self.planning_budget = planning_budget
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
        if len(simulator.robots) != 1:
            raise ValueError("Cube stacking requires one robot")
        self.robot = next(iter(simulator.robots.values()))
        self.robot_type = self.robot.robot_type
        if self.robot_type == "panda":
            controller = create_controller(
                "position",
                self.robot,
                gravity_compensation=[True] * 7 + [False],
                frame="grasp",
            )
        elif self.robot_type == "forte":
            controller = create_controller("pd", self.robot, frame="grasp")
        else:
            raise ValueError("Cube stacking requires panda or forte")
        self.robot.change_controller(controller)
        self._plan = self._make_plan()
        self.stage = 0
        self.stage_ticks = 0
        self.max_lift = self.starts[:, 2].copy()
        self._stable_since = None
        self._stage_path = None
        self._path_vertex = 1
        self._path_progress = 0.0
        self._planning_epoch = 0
        self.failed = False
        self.failure_reason = None
        simulator.target_updater = self.update

    def _make_plan(self):
        if self.robot_type == "forte":
            return self._make_forte_plan()
        seed = self.robot.joint_state.qpos[:7].copy()
        state = self.robot.state
        rotation = self.simulator.data.site_xmat[state.site_id("grasp")].reshape(3, 3).copy()
        plan = []
        for index in np.argsort(self.goals[:, 2]):
            start = self.starts[index]
            goal = self.goals[index]
            for label, xyz, finger, wait in (
                ("above_pick", (start[0], start[1], start[2] + 0.23), 0.04, 100),
                ("pick", (start[0], start[1], start[2] + 0.02), 0.04, 100),
                ("close", (start[0], start[1], start[2] + 0.02), 0.011, 1800),
                ("lift", (start[0], start[1], start[2] + 0.25), 0.011, 100),
                ("above_place", (goal[0], goal[1], goal[2] + 0.26), 0.011, 100),
                ("place", (goal[0], goal[1], goal[2] + 0.035), 0.011, 100),
                ("release", (goal[0], goal[1], goal[2] + 0.035), 0.04, 900),
                ("retract", (goal[0], goal[1], goal[2] + 0.26), 0.04, 100),
            ):
                pose = Transform(rotation=rotation, translation=xyz)
                seed = state.solve_ik(pose, frame="grasp", seed=seed)
                plan.append((f"cube{index}:{label}", np.r_[seed, finger], wait))
        return plan

    def _make_forte_plan(self):
        """Plan top-down grasps using the pad-centered grasp site."""
        seed = self.robot.joint_state.qpos[:7].copy()
        state = self.robot.state
        rotation = Rotation.from_euler("x", np.pi)
        plan = []
        for index in np.argsort(self.goals[:, 2]):
            start, goal = self.starts[index], self.goals[index]
            for label, xyz, finger, wait in (
                ("above_pick", (start[0], start[1], start[2] + 0.11), 0.0, 180),
                ("pick", (start[0], start[1], start[2] + 0.01), 0.0, 180),
                ("close", (start[0], start[1], start[2] + 0.01), -0.02, 550),
                ("lift", (start[0], start[1], start[2] + 0.11), -0.02, 180),
                ("above_place", (goal[0], goal[1], goal[2] + 0.11), -0.02, 180),
                ("place", (goal[0], goal[1], goal[2] + 0.01), -0.02, 180),
                ("release", (goal[0], goal[1], goal[2] + 0.01), 0.0, 500),
                ("retract", (goal[0], goal[1], goal[2] + 0.11), 0.0, 180),
            ):
                pose = Transform(rotation=rotation, translation=xyz)
                seed = state.solve_ik(pose, frame="grasp", seed=seed)
                plan.append((f"cube{index}:{label}", np.r_[seed, finger], wait))
        return plan

    @property
    def stage_name(self):
        return self._plan[self.stage][0] if self.stage < len(self._plan) else "settle"

    def update(self, simulator: Simulator):
        if self.method == "sampling":
            self._update_sampling(simulator)
        else:
            self._update_heuristic(simulator)

    def _update_heuristic(self, simulator: Simulator):
        if self.status().released_stable_stack:
            simulator.stop()
        if self.stage >= len(self._plan):
            return
        stage_name, command, dwell = self._plan[self.stage]
        current = self.robot.joint_state.qpos
        previous = self.robot.target.position
        # Limit target changes so the arm and fingers move continuously.
        rate = 0.004
        if self.robot_type == "forte" and stage_name.split(":", 1)[1] in (
            "lift",
            "above_place",
            "place",
        ):
            rate = 0.0004
        delta = np.clip(command[:7] - previous[:7], -rate, rate)
        if self.robot_type == "panda":
            finger_delta = np.clip(command[7] - previous[7], -0.0004, 0.0004)
            finger_target = previous[7] + finger_delta
            self.robot.target = ControlTarget(np.r_[previous[:7] + delta, finger_target])
        else:
            finger_previous = self.robot.controller.gripper_target
            finger_delta = np.clip(command[7] - finger_previous, -0.0002, 0.0002)
            finger_target = finger_previous + finger_delta
            self.robot.controller.set_gripper_target(finger_target)
            self.robot.target = ControlTarget(previous[:7] + delta)
        self.stage_ticks += 1
        close_enough = np.max(np.abs(command[:7] - current[:7])) < 0.016
        # A cube prevents the fingers from reaching a commanded closing gap.
        finger_close = abs(command[7] - finger_target) < 0.001
        physical_ready = True
        if self.robot_type == "forte" and stage_name.endswith(":close"):
            index = int(stage_name.split(":", 1)[0][4:])
            cube_geom = f"cube{index}/object_0"
            pad_names = tuple(f"{self.robot.name}/gripper_{side}_pad" for side in ("left", "right"))
            contacts = {
                frozenset(
                    (
                        self.simulator.model.geom(contact.geom1).name,
                        self.simulator.model.geom(contact.geom2).name,
                    )
                )
                for contact in self.simulator.data.contact
            }
            physical_ready = all(frozenset((cube_geom, pad)) in contacts for pad in pad_names)
        if self.robot_type == "forte" and stage_name.endswith(":lift"):
            index = int(stage_name.split(":", 1)[0][4:])
            physical_ready = (
                self.simulator.data.body(f"cube{index}/object_0").xpos[2]
                > self.starts[index, 2] + 0.04
            )
        if close_enough and finger_close and physical_ready and self.stage_ticks >= dwell:
            self.stage += 1
            self.stage_ticks = 0

    def _plan_stage(self, stage_name: str, goal: np.ndarray) -> np.ndarray | None:
        """Search the current scene for one joint-space stage transition."""
        from mujoco_lab.planning import (
            PRM,
            RRT,
            GoalBiasedSampler,
            PRMConfig,
            RRTConfig,
            RRTConnect,
            RRTConnectConfig,
        )
        from mujoco_lab.planning.collision.manipulation import CubeStackCollisionChecker

        cube_index = int(stage_name.split(":", 1)[0][4:])
        label = stage_name.split(":", 1)[1]
        checker = CubeStackCollisionChecker(
            self.robot,
            stage_name,
            support_geom=self._supports[cube_index] if label == "place" else None,
        )
        start = self.simulator.data.qpos[checker.qpos_indices].copy()
        destination = np.asarray(goal[:7], dtype=float)
        bounds = [tuple(row) for row in checker.bounds]
        stage_seed = self.seed + self._planning_epoch
        self._planning_epoch += 1
        if self.planner == "rrt_connect":
            search = RRTConnect(
                start,
                destination,
                bounds,
                checker,
                RRTConnectConfig(
                    max_iterations=self.planning_budget,
                    step_size=0.2,
                    goal_tolerance=0.04,
                    seed=stage_seed,
                ),
            )
        elif self.planner == "rrt":
            search = RRT(
                start,
                destination,
                bounds,
                checker,
                RRTConfig(
                    max_iterations=self.planning_budget,
                    step_size=0.2,
                    goal_tolerance=0.04,
                    goal_bias=0.8,
                    seed=stage_seed,
                ),
            )
        else:
            search = PRM(
                start,
                destination,
                bounds,
                checker,
                PRMConfig(
                    sample_number=self.planning_budget,
                    max_retries=2,
                    radius=2.0,
                    sampler=GoalBiasedSampler,
                    goal_bias=0.3,
                    seed=stage_seed,
                ),
            )
        result = search.plan()
        if result is None:
            return None
        vertices = np.asarray([node.state for node in result], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 7:
            raise RuntimeError("Planner returned an invalid arm path")
        if len(vertices) == 1:
            vertices = np.vstack((vertices, destination))
        if not np.allclose(vertices[0], start, atol=1e-9, rtol=0):
            raise RuntimeError("Planner path does not start at measured state")
        if not np.allclose(vertices[-1], destination, atol=1e-9, rtol=0):
            raise RuntimeError("Planner path does not end at stage target")
        return vertices

    def _has_physical_grasp(self, cube_index: int) -> bool:
        """Require contact from both physical fingers before carrying a cube."""
        model, data = self.simulator.model, self.simulator.data
        cube_geom = model.geom(f"cube{cube_index}/object_0").id
        sides = set()
        for contact in data.contact:
            if contact.geom1 == cube_geom:
                other = int(contact.geom2)
            elif contact.geom2 == cube_geom:
                other = int(contact.geom1)
            else:
                continue
            if self.robot_type == "forte":
                for side in ("left", "right"):
                    if model.geom(other).name == f"{self.robot.name}/gripper_{side}_pad":
                        sides.add(side)
            else:
                body = model.body(int(model.geom_bodyid[other])).name
                for side in ("left", "right"):
                    if body == f"{self.robot.name}/panda_{side}finger":
                        sides.add(side)
        return sides == {"left", "right"}

    def _update_sampling(self, simulator: Simulator):
        if self.failed:
            simulator.stop()
            return
        if self.status().released_stable_stack:
            simulator.stop()
        if self.stage >= len(self._plan):
            return
        stage_name, command, dwell = self._plan[self.stage]
        label = stage_name.split(":", 1)[1]
        cube_index = int(stage_name.split(":", 1)[0][4:])
        motion = label not in ("close", "release")
        if motion and self._stage_path is None:
            if label == "above_pick":
                actual = simulator.data.body(f"cube{cube_index}/object_0").xpos
                if np.linalg.norm(actual - self.starts[cube_index]) > 0.01:
                    self.failed = True
                    self.failure_reason = (
                        f"cube{cube_index} moved more than 1 cm from its planned pick pose"
                    )
                    simulator.stop()
                    return
            if label in ("lift", "above_place", "place") and not self._has_physical_grasp(
                cube_index
            ):
                self.failed = True
                self.failure_reason = f"No two-finger physical grasp for {stage_name}"
                simulator.stop()
                return
            self._stage_path = self._plan_stage(stage_name, command)
            self._path_vertex = 1
            self._path_progress = 0.0
            if self._stage_path is None:
                self.failed = True
                self.failure_reason = f"No {self.planner} route for {stage_name}"
                simulator.stop()
                return

        current = self.robot.joint_state.qpos
        previous = self.robot.target.position
        rate = 0.004
        if self.robot_type == "forte" and label in ("lift", "above_place", "place"):
            rate = 0.0004
        path_complete = not motion
        if motion and self._path_vertex < len(self._stage_path):
            source = self._stage_path[self._path_vertex - 1]
            destination = self._stage_path[self._path_vertex]
            edge = destination - source
            length = float(np.max(np.abs(edge)))
            self._path_progress = min(1.0, self._path_progress + rate / max(length, rate))
            arm_target = source + self._path_progress * edge
            if self._path_progress == 1.0 and np.max(np.abs(destination - current[:7])) < 0.016:
                self._path_vertex += 1
                self._path_progress = 0.0
        else:
            arm_target = command[:7]
            path_complete = True
        if motion and self._path_vertex == len(self._stage_path):
            path_complete = True

        if self.robot_type == "panda":
            finger_delta = np.clip(command[7] - previous[7], -0.0004, 0.0004)
            finger_target = previous[7] + finger_delta
            self.robot.target = ControlTarget(np.r_[arm_target, finger_target])
        else:
            finger_previous = self.robot.controller.gripper_target
            finger_delta = np.clip(command[7] - finger_previous, -0.0002, 0.0002)
            finger_target = finger_previous + finger_delta
            self.robot.controller.set_gripper_target(finger_target)
            self.robot.target = ControlTarget(arm_target)
        self.stage_ticks += 1
        close_enough = np.max(np.abs(command[:7] - current[:7])) < 0.016
        finger_close = abs(command[7] - finger_target) < 0.001
        physical_ready = True
        if self.robot_type == "forte" and label == "close":
            cube_index = int(stage_name.split(":", 1)[0][4:])
            cube_geom = f"cube{cube_index}/object_0"
            contacts = {
                frozenset(
                    (
                        simulator.model.geom(contact.geom1).name,
                        simulator.model.geom(contact.geom2).name,
                    )
                )
                for contact in simulator.data.contact
            }
            physical_ready = all(
                frozenset((cube_geom, f"{self.robot.name}/gripper_{side}_pad")) in contacts
                for side in ("left", "right")
            )
        if self.robot_type == "forte" and label == "lift":
            cube_index = int(stage_name.split(":", 1)[0][4:])
            physical_ready = (
                simulator.data.body(f"cube{cube_index}/object_0").xpos[2]
                > self.starts[cube_index, 2] + 0.04
            )
        if (
            path_complete
            and close_enough
            and finger_close
            and physical_ready
            and self.stage_ticks >= dwell
        ):
            self.stage += 1
            self.stage_ticks = 0
            self._stage_path = None
        elif self.stage_ticks > max(6000, dwell + 2000):
            self.failed = True
            self.failure_reason = f"Timed out executing {stage_name}"
            simulator.stop()

    def status(self) -> StackStatus:
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
        self.max_lift = np.maximum(self.max_lift, centers[:, 2])
        released = self.stage >= len(self._plan) and (
            self.robot.joint_state.qpos[7] > 0.033
            if self.robot_type == "panda"
            else self.robot.joint_state.qpos[7] > -0.002
        )
        aligned = np.all(np.linalg.norm(centers[:, :2] - self.goals[:, :2], axis=1) < 0.018)
        heights = np.all(np.abs(centers[:, 2] - self.goals[:, 2]) < 0.012)
        still = np.max(velocities) < 0.025
        contacts = {
            frozenset((model.geom(contact.geom1).name, model.geom(contact.geom2).name))
            for contact in data.contact
        }
        support_contacts = all(
            frozenset((self._supports[i], f"cube{i}/object_0")) in contacts
            for i in range(self.cubes)
        )
        touching_robot = any(
            any(name.startswith(self.robot.name + "/") for name in pair)
            and any(name == f"cube{i}/object_0" for name in pair for i in range(self.cubes))
            for pair in contacts
        )
        stable = (
            released and aligned and heights and still and support_contacts and not touching_robot
        )
        if stable:
            if self._stable_since is None:
                self._stable_since = data.time
        else:
            self._stable_since = None
        return StackStatus(
            distances,
            bool(np.all(distances <= 0.04)),
            self._stable_since is not None and data.time - self._stable_since >= 0.5,
            support_contacts,
            centers[:, 2].copy(),
        )

    def reset(self):
        self.simulator.reset()
        self.stage = 0
        self.stage_ticks = 0
        self.max_lift[:] = self.starts[:, 2]
        self._stable_since = None
        self._stage_path = None
        self._path_vertex = 1
        self._path_progress = 0.0
        self._planning_epoch = 0
        self.failed = False
        self.failure_reason = None
