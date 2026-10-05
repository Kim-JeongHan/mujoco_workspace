"""Command-line entry point for physical cube stacking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import CubeCount, RobotName
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import (
    CubeStackExpert,
    CubeStackTask,
)
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control import create_controller
from mujoco_lab.planning import (
    PlannerConfig,
    default_planning,
    planner_name,
)
from mujoco_lab.rendering.camera import create_free_camera
from mujoco_lab.simulator_manager import SimulatorManager


@dataclass
class Config:
    """Choose the stack size, run mode, and optional image or video output."""

    cubes: CubeCount = 2
    robot: RobotName = "panda"
    environment: Literal["table_shelf", "warehouse"] = "table_shelf"
    method: Literal["heuristic", "sampling"] = "heuristic"
    planning: PlannerConfig = field(default_factory=default_planning)
    headless: bool = False
    steps: int | None = None
    output: Path | None = None
    video: Path | None = None  # Record the direct simulator run as MP4.
    fps: int = 30  # Saved video frames per simulated second.
    video_width: int = 640  # MP4 frame width in pixels.
    video_height: int = 480  # MP4 frame height in pixels.
    camera_azimuth: float = 60.0  # Initial oblique view; the viewer can adjust it.
    camera_elevation: float = -25.0  # Degrees below the horizontal plane.
    camera_distance: float = 1.25  # View distance from the cube workspace in meters.


def main() -> None:
    config = tyro.cli(Config, description="Physical cube stacking with Panda or Forte")
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    steps = config.steps
    if steps is None:
        if config.method == "sampling":
            steps = 90000 if config.robot == "forte" else 60000
        else:
            steps = 30000 if config.robot == "forte" else 22000
    if steps < 0:
        logger.error("--steps must be zero or greater", exit_code=2)
    robot_config = load_robot_config(config.robot)
    simulator = Simulator(
        create_cube_stack(config.cubes, environment=config.environment),
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
    )
    robot = simulator.robots[config.robot]
    controller = create_controller(robot, robot_config.controller)
    robot.change_controller(controller)
    task = CubeStackTask(simulator, config.cubes)
    recipe = load_cube_recipe(config.robot)
    if config.method == "sampling":
        expert = CubeStackExpert(
            task,
            recipe=recipe,
            method="sampling",
            planning=config.planning,
        )
        planner_label = planner_name(config.planning)
    else:
        planner_label = "none"
        expert = CubeStackExpert(
            task,
            recipe=recipe,
            method="heuristic",
        )
    simulator.target_updater = expert.update

    name = "cube_stack"
    manager.add_simulator(name, simulator)
    try:
        if config.video is not None:
            center = (task.starts.mean(axis=0) + task.goals.mean(axis=0)) / 2
            center[2] += 0.12
            camera = create_free_camera(
                lookat=center,
                azimuth=config.camera_azimuth,
                elevation=config.camera_elevation,
                distance=config.camera_distance,
            )
        if config.headless:
            if config.video is None:
                simulator.run_steps(steps)
            else:
                video_path = manager.save_video(
                    name,
                    steps,
                    config.video,
                    camera=camera,
                    fps=config.fps,
                    width=config.video_width,
                    height=config.video_height,
                )
                logger.info(f"Saved video: {video_path}")
        else:
            if config.video is None:
                manager.show(name)
            else:
                video_path = manager.show(
                    name,
                    camera=camera,
                    video=config.video,
                    fps=config.fps,
                    width=config.video_width,
                    height=config.video_height,
                    steps=steps,
                )
                if video_path is None:
                    logger.info("Preview closed before recording; no new video was recorded")
                else:
                    logger.info(f"Saved video: {video_path}")
        status = task.status()
        failure = expert.failure_reason
        if config.headless and not status.released_stable_stack and failure is None:
            completed_steps = round(simulator.data.time / simulator.dt)
            failure = (
                f"step budget exhausted at {expert.get_stage_name()} after {completed_steps} steps"
            )
        if config.output is not None:
            manager.save_frame(name, config.output)
        logger.info(
            f"robot={config.robot} cubes={config.cubes} method={config.method} "
            f"planner={planner_label} "
            f"simulated={simulator.data.time:.3f}s "
            f"stage={expert.get_stage_name()} ogbench_goal={status.ogbench_success} "
            f"released_stable_stack={status.released_stable_stack} "
            f"support_contacts={status.support_contacts} "
            f"goal_distances_m={status.goal_distances.round(4).tolist()} "
            f"max_heights_m={task.max_lift.round(3).tolist()} "
            f"failure={failure}"
        )
        if config.headless and not status.released_stable_stack:
            raise SystemExit(1)
    finally:
        manager.remove_simulator(name)
