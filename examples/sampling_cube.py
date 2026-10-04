"""Stack cubes with a sampling planner: uv run python examples/sampling_cube.py --headless."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control import create_controller
from mujoco_lab.planning import (
    PRM,
    RRG,
    RRT,
    GoalBiasedSampler,
    PRMConfig,
    PRMStar,
    PRMStarConfig,
    RRGConfig,
    RRTConfig,
    RRTConnect,
    RRTConnectConfig,
    RRTStar,
    RRTStarConfig,
)
from mujoco_lab.rendering.camera import create_free_camera


@dataclass
class Config:
    """Choose a planner and optionally record the physical cube stack."""

    planner: Literal["rrt-connect", "rrt", "prm", "rrt-star", "prm-star", "rrg"] = "rrt-connect"
    robot: Literal["forte", "panda"] = "forte"
    cubes: int = 2
    environment: Literal["table_shelf", "warehouse"] = "table_shelf"
    headless: bool = False
    steps: int = 90_000
    video: Path | None = None
    fps: int = 30
    video_width: int = 640
    video_height: int = 480
    camera_azimuth: float = 60.0
    camera_elevation: float = -25.0
    camera_distance: float = 1.25


class Planner:
    """Run the selected low-level planner for each cube-stack stage."""

    def __init__(self, name: str) -> None:
        self.name = name.replace("-", "_")
        self.seed = 7

        choices = {
            "rrt-connect": (
                RRTConnect,
                RRTConnectConfig(
                    max_iterations=500, step_size=0.2, goal_tolerance=0.04, seed=self.seed
                ),
            ),
            "rrt": (
                RRT,
                RRTConfig(
                    max_iterations=500,
                    step_size=0.2,
                    goal_tolerance=0.04,
                    goal_bias=0.8,
                    seed=self.seed,
                ),
            ),
            "prm": (
                PRM,
                PRMConfig(
                    sample_number=100,
                    max_retries=2,
                    radius=2.0,
                    sampler=GoalBiasedSampler,
                    goal_bias=0.3,
                    seed=self.seed,
                ),
            ),
            "rrt-star": (
                RRTStar,
                RRTStarConfig(
                    max_iterations=500,
                    step_size=0.2,
                    goal_tolerance=0.04,
                    goal_bias=0.1,
                    radius_gain=1.0,
                    return_first_solution=True,
                    seed=self.seed,
                ),
            ),
            "prm-star": (
                PRMStar,
                PRMStarConfig(
                    sample_number=120,
                    max_retries=3,
                    step_size=0.2,
                    goal_tolerance=0.15,
                    goal_bias=0.05,
                    radius_gain=5.0,
                    seed=self.seed,
                ),
            ),
            "rrg": (
                RRG,
                RRGConfig(
                    max_iterations=5000,
                    radius_gain=0.5,
                    step_size=0.15,
                    goal_tolerance=0.005,
                    goal_bias=0.1,
                    seed=self.seed,
                ),
            ),
        }
        self.planner_type, self.config = choices[self.name.replace("_", "-")]

    def plan(self, start, goal, bounds, collision_checker, *, seed):
        config = self.config.model_copy(update={"seed": seed})
        nodes = self.planner_type(start, goal, bounds, collision_checker, config).plan()
        if nodes is None:
            return None
        path = np.asarray([node.state for node in nodes], dtype=float)
        # RRG may stop within its goal tolerance; the task needs the exact endpoint.
        if self.name == "rrg" and not np.array_equal(path[-1], goal):
            if not collision_checker.is_path_collision_free(path[-1], goal):
                return None
            path = np.vstack((path, goal))
        return path


def main() -> None:
    config = tyro.cli(Config, description="Stack cubes with a sampling planner")
    manager = SimulatorManager()
    if config.steps < 0:
        manager.logger.error("--steps must be zero or greater", exit_code=2)

    robot_config = load_robot_config(config.robot)
    simulator = Simulator(
        create_cube_stack(config.cubes, environment=config.environment),
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
    )
    robot = simulator.robots[config.robot]
    controller = create_controller(robot, robot_config.controller)
    robot.change_controller(controller)
    task = CubeStackTask(simulator, config.cubes)
    planner = Planner(config.planner)
    recipe = load_cube_recipe(config.robot)
    expert = CubeStackExpert(
        task,
        recipe=recipe,
        method="sampling",
        planner=planner,
    )
    simulator.target_updater = expert.update

    name = "sampling_cube"
    manager.add_simulator(name, simulator)
    if config.headless and config.video is None:
        simulator.run_steps(config.steps)
    else:
        center = (task.starts.mean(axis=0) + task.goals.mean(axis=0)) / 2
        center[2] += 0.12
        camera = create_free_camera(
            lookat=center,
            azimuth=config.camera_azimuth,
            elevation=config.camera_elevation,
            distance=config.camera_distance,
        )
        if config.video is None:
            manager.show(name, camera=camera)
        elif config.headless:
            manager.save_video(
                name,
                config.steps,
                config.video,
                camera=camera,
                fps=config.fps,
                width=config.video_width,
                height=config.video_height,
            )
        else:
            manager.show(
                name,
                camera=camera,
                video=config.video,
                steps=config.steps,
                fps=config.fps,
                width=config.video_width,
                height=config.video_height,
            )

    status = task.status()
    manager.logger.info(
        f"planner={planner.name} robot={config.robot} simulated={simulator.data.time:.3f}s "
        f"released_stable_stack={status.released_stable_stack} "
        f"support_contacts={status.support_contacts} "
        f"goal_distances_m={status.goal_distances.round(4).tolist()} "
        f"stage={expert.get_stage_name()} failure={expert.failure_reason}"
    )
    if config.headless and not status.released_stable_stack:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
