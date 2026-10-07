"""Run cube stacking: uv run python examples/cube_task.py --method sampling --headless."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_cube_stack
from mujoco_lab.assets import CubeCount, RobotName
from mujoco_lab.assets.loader import load_robot_config, randomize_cube_positions
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control import create_controller
from mujoco_lab.planning import PlannerConfig, default_planning, planner_name


@dataclass
class Config:
    """Run heuristic or sampling cube stacking."""

    cubes: CubeCount = 1
    seed: int = 50
    xy_range: float = 0.02
    min_gap: float = 0.01
    cube_yaw_range_degrees: float = 45
    robot: RobotName = "forte"
    environment: Literal["table_shelf", "warehouse"] = "table_shelf"
    method: Literal["heuristic", "sampling"] = "heuristic"
    planning: PlannerConfig = field(default_factory=default_planning)
    headless: bool = False
    steps: int | None = None
    output: Path | None = None


def main() -> None:
    config = tyro.cli(Config, description="Run heuristic or sampling cube stacking")
    steps = config.steps
    if steps is None:
        if config.method == "sampling":
            steps = 90000 if config.robot == "forte" else 60000
        else:
            steps = 30000 if config.robot == "forte" else 22000
    if steps < 0:
        raise ValueError("steps must be zero or greater")

    robot_config = load_robot_config(config.robot)
    scene = create_cube_stack(config.cubes, environment=config.environment)
    randomize_cube_positions(
        scene,
        xy_range=config.xy_range,
        min_gap=config.min_gap,
        yaw_range_degrees=config.cube_yaw_range_degrees,
        seed=config.seed,
    )
    simulator = Simulator(
        scene,
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
    )
    robot = simulator.robots[config.robot]
    robot.change_controller(create_controller(robot, robot_config.controller))
    task = CubeStackTask(simulator, config.cubes)
    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe(config.robot),
        method=config.method,
        planning=config.planning if config.method == "sampling" else None,
    )
    simulator.target_updater = expert.update

    manager = SimulatorManager()
    name = "cube_stack"
    manager.add_simulator(name, simulator)
    try:
        if config.headless:
            simulator.run_steps(steps)
        else:
            manager.show(name)
        if config.output is not None:
            manager.save_frame(name, config.output)

        status = task.status()
        failure = expert.failure_reason
        if config.headless and not status.released_stable_stack and failure is None:
            failure = f"step budget exhausted during {expert.get_stage_name()}"
        planner_label = planner_name(config.planning) if config.method == "sampling" else "none"
        manager.logger.info(
            f"robot={config.robot} cubes={config.cubes} method={expert.method} "
            f"planner={planner_label} simulated={simulator.data.time:.3f}s "
            f"stage={expert.get_stage_name()} "
            f"released_stable_stack={status.released_stable_stack} "
            f"support_contacts={status.support_contacts} "
            f"goal_distances_m={status.goal_distances.round(4).tolist()} "
            f"failure={failure}"
        )
        if config.headless and not status.released_stable_stack:
            raise SystemExit(1)
    finally:
        manager.remove_simulator(name)


if __name__ == "__main__":
    main()
