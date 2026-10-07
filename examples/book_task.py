"""Run book insertion: uv run python examples/book_task.py --book medium --method heuristic."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_book_insertion
from mujoco_lab.assets import BookType, RobotName
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import BookInsertionExpert, BookTask
from mujoco_lab.behaviors.book import create_book_controller
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.planning import (
    PlannerConfig,
    default_planning,
    planner_name,
)
from mujoco_lab.rendering.camera import create_free_camera


@dataclass
class Config:
    """Run sampling or heuristic book insertion."""

    book: BookType = "medium"
    robot: RobotName = "forte"
    method: Literal["sampling", "heuristic"] = "heuristic"
    planning: PlannerConfig = field(default_factory=default_planning)
    headless: bool = False
    steps: int = 60000
    output: Path | None = None


def main() -> None:
    config = tyro.cli(Config, description="Run a centered book grasp and shelf insertion")
    if config.steps < 0:
        raise ValueError("steps must be zero or greater")
    robot_config = load_robot_config(config.robot)
    sim = Simulator(
        create_book_insertion(config.book),
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
    )
    robot = sim.robots[config.robot]
    robot.change_controller(create_book_controller(robot, robot_config.controller))
    task = BookTask(sim)
    recipe = load_book_recipe(config.robot)
    expert = BookInsertionExpert(
        task,
        recipe=recipe,
        method=config.method,
        planning=config.planning if config.method == "sampling" else None,
    )
    sim.target_updater = expert.update
    manager = SimulatorManager()
    camera = create_free_camera(
        lookat=np.array([0.35, 0.15, 1.1]), azimuth=60, elevation=-20, distance=2.5
    )
    name = "books"
    manager.add_simulator(name, sim)
    if config.headless:
        sim.run_steps(config.steps)
    else:
        manager.show(name, camera=camera)
    if config.output is not None:
        manager.save_frame(name, config.output, camera=camera)
    book = sim.model.body("book")
    size = 2 * sim.model.geom("book/collision").size
    manager.logger.info(
        f"book={config.book} size_m={size.tolist()} mass_kg={book.mass[0]} "
        f"inertia_kg_m2={book.inertia.tolist()} center_m={sim.data.body('book').xpos.tolist()} "
        f"simulated={sim.data.time:.3f}s"
    )
    status = task.status()
    failure = expert.failure_reason
    planner_label = planner_name(config.planning) if config.method == "sampling" else "none"
    if config.headless and not status.released_stable and failure is None:
        failure = f"step budget exhausted during {expert.get_stage_name()}"
    manager.logger.info(
        f"stage={expert.get_stage_name()} method={expert.method} "
        f"planner={planner_label} "
        f"released_stable={status.released_stable} "
        f"position_error_m={status.position_error:.4f} "
        f"rotation_error_rad={status.rotation_error:.4f} "
        f"lift_m={task.max_height - task.start_center[2]:.4f} "
        f"failure={failure}"
    )
    if config.headless and not status.released_stable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
