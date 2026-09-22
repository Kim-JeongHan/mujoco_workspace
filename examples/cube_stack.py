"""Physically stack OGBench-layout cubes with Panda or Forte.

Example: uv run python examples/cube_stack.py --cubes 2 --headless
Sampling checks a frozen scene at discrete points per stage, then executes
the planned joint vertices using the native robot controller.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from mujoco_lab import SimulatorManager, create_cube_stack
from mujoco_lab.tasks import CubeStackTask


@dataclass
class Config:
    """Choose the stack size, run mode, and optional PNG output."""

    cubes: int = 2
    robot: str = "panda"
    environment: str = "table_shelf"
    method: Literal["heuristic", "sampling"] = "heuristic"
    planner: Literal["rrt_connect", "rrt", "prm"] = "rrt_connect"
    seed: int = 7
    planning_budget: int = 500
    headless: bool = False
    steps: int | None = None
    output: Path | None = None


def main() -> None:
    config = tyro.cli(Config)
    steps = config.steps
    if steps is None:
        if config.method == "sampling":
            steps = 90000 if config.robot == "forte" else 60000
        else:
            steps = 30000 if config.robot == "forte" else 22000
    if steps < 0:
        raise SystemExit("--steps must be zero or greater")
    try:
        simulator = create_cube_stack(
            config.cubes,
            environment=config.environment,
            robot=config.robot,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        task = CubeStackTask(
            simulator,
            config.cubes,
            method=config.method,
            planner=config.planner,
            seed=config.seed,
            planning_budget=config.planning_budget,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    manager = SimulatorManager.get_instance()
    name = "cube_stack"
    manager.add_simulator(name, simulator)
    try:
        if config.headless:
            simulator.run_steps(steps)
        else:
            manager.show(name)
        status = task.status()
        failure = task.failure_reason
        if config.headless and not status.released_stable_stack and failure is None:
            completed_steps = round(simulator.data.time / simulator.dt)
            failure = f"step budget exhausted at {task.stage_name} after {completed_steps} steps"
        if config.output is not None:
            manager.save_frame(name, config.output)
        print(
            f"robot={config.robot} cubes={config.cubes} method={config.method} "
            f"planner={config.planner if config.method == 'sampling' else 'none'} "
            f"simulated={simulator.data.time:.3f}s "
            f"stage={task.stage_name} ogbench_goal={status.ogbench_success} "
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


if __name__ == "__main__":
    main()
