"""Plan and execute a short arm motion in a MuJoCo scene."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro

from mujoco_lab import ROBOT_NAMES, RobotSpec, Simulator, SimulatorManager, create_environment
from mujoco_lab.control import create_controller
from mujoco_lab.planning import (
    PRM,
    RRT,
    MuJoCoCollisionChecker,
    PRMConfig,
    RRTConfig,
    RRTConnect,
    RRTConnectConfig,
)
from mujoco_lab.planning.follower import WaypointFollower
from mujoco_lab.rendering.path_overlay import PathOverlay


@dataclass
class Config:
    """Plan from home and move the selected robot through the checked path."""

    robot: Literal[ROBOT_NAMES] = "forte"
    planner: Literal["rrt", "rrt_connect", "prm"] = "rrt"
    output: Path | None = None
    image: Path | None = None
    seed: int = 7
    headless: bool = False
    steps: int = 6000


def main() -> None:
    config = tyro.cli(Config, description="Plan and execute a joint-space arm path")
    if config.steps <= 0:
        raise SystemExit("--steps must be positive")
    simulator = Simulator(
        create_environment("empty"), robots=[RobotSpec(config.robot, config.robot)]
    )
    robot = simulator.robots[config.robot]
    selection = (
        {"frame": "grasp"}
        if config.robot in ("panda", "forte")
        else {"joint_names": robot.state.joint_names}
    )

    # A finite local window is required for Forte's continuous joints. Limited
    # joints are clipped to their actual hardware ranges, including Panda/UR.
    joint_ids = [
        robot.model.joint(robot.prefix + name).id
        for name in (
            robot.state.joint_names[:7]
            if config.robot in ("panda", "forte")
            else robot.state.joint_names
        )
    ]
    current = simulator.data.qpos[robot.model.jnt_qposadr[joint_ids]].copy()
    bounds = []
    for joint_id, position in zip(joint_ids, current, strict=True):
        lower, upper = position - 0.35, position + 0.35
        if robot.model.jnt_limited[joint_id]:
            hardware = robot.model.jnt_range[joint_id]
            lower, upper = max(lower, hardware[0]), min(upper, hardware[1])
        bounds.append((lower, upper))
    checker = MuJoCoCollisionChecker(robot, **selection, bounds=bounds)
    start = simulator.data.qpos[checker.qpos_indices].copy()
    goal = start.copy()
    goal[1] += 0.25

    if config.planner == "rrt":
        planner = RRT(
            start,
            goal,
            bounds,
            checker,
            RRTConfig(
                max_iterations=300,
                step_size=0.2,
                goal_tolerance=0.04,
                goal_bias=0.2,
                seed=config.seed,
            ),
        )
    elif config.planner == "rrt_connect":
        planner = RRTConnect(
            start,
            goal,
            bounds,
            checker,
            RRTConnectConfig(max_iterations=300, step_size=0.2, seed=config.seed),
        )
    else:
        planner = PRM(
            start,
            goal,
            bounds,
            checker,
            PRMConfig(sample_number=100, max_retries=2, radius=0.7, seed=config.seed),
        )
    path = planner.plan()
    if path is None:
        raise SystemExit("No path found for this frozen scene and planning window")
    waypoints = np.asarray([node.state for node in path])
    print(f"{config.robot} {config.planner}: {len(path)} joint-space waypoints")
    print("Joint order:", ", ".join(checker.joint_names))
    print(waypoints)
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with config.output.open("wb") as file:
            np.savez(file, joint_names=checker.joint_names, waypoints=waypoints)
        print(f"Saved {config.output}")

    robot.change_controller(
        create_controller("pd" if config.robot == "forte" else "position", robot)
    )
    follower = WaypointFollower(
        robot,
        checker.joint_names,
        waypoints,
        max_steps=config.steps,
        stop_on_finish=config.headless,
    )
    frame = "grasp" if config.robot in ("panda", "forte") else "tool0"
    overlay = PathOverlay(robot, checker.joint_names, waypoints, frame=frame)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = overlay.planned_xyz.mean(axis=0)
    if config.robot.startswith("ur"):
        camera.lookat[2] = 0.5
        camera.distance = 1.9
    else:
        camera.lookat[2] -= 0.15
        camera.distance = 1.3
    camera.azimuth = 135
    camera.elevation = -25
    announced = False

    def report() -> None:
        print(follower.result())
        if overlay.current_xyz is not None:
            travel = np.linalg.norm(overlay.current_xyz - overlay.planned_xyz[0])
            error = np.linalg.norm(overlay.current_xyz - overlay.planned_xyz[-1])
            print(f"Measured endpoint travel {travel:.4f} m; goal error {error:.4f} m")

    def update(sim: Simulator) -> None:
        nonlocal announced
        follower(sim)
        overlay.record(sim.data)
        if not config.headless and follower.complete and not announced:
            print("Reached every waypoint. The viewer remains open; close it to exit.")
            announced = True

    simulator.target_updater = update
    manager = SimulatorManager.get_instance()
    if config.headless:
        simulator.run_steps(config.steps)
        follower.check_arrival(simulator)
        if not follower.complete and follower.steps >= config.steps:
            follower.failed = True
        overlay.record(simulator.data)
        robot.update_state()
        report()
        if config.image is not None:
            name = "planning"
            manager.add_simulator(name, simulator)
            try:
                image_path = manager.save_frame(name, config.image, draw=overlay, camera=camera)
                print(f"Saved {image_path}")
            finally:
                manager.remove_simulator(name)
        if not follower.complete:
            raise SystemExit("Robot did not reach every planned waypoint")
    else:
        name = "planning"
        manager.add_simulator(name, simulator)
        try:
            print("Close the viewer after inspecting the planned path and actual trail.")
            manager.show(name, draw=overlay, camera=camera)
        finally:
            manager.remove_simulator(name)
        robot.update_state()
        overlay.record(simulator.data)
        report()
        if not follower.complete:
            raise SystemExit("Robot did not reach every planned waypoint")


if __name__ == "__main__":
    main()
