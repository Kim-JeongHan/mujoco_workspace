"""Select a manipulator: uv run python examples/manipulator.py --robot panda."""

from dataclasses import dataclass
from typing import Literal

import tyro

from mujoco_lab import (
    ENVIRONMENT_NAMES,
    ROBOT_NAMES,
    RobotSpec,
    Simulator,
    SimulatorManager,
    create_environment,
)
from mujoco_lab.control import CONTROLLER_NAMES, create_controller, demo_target_updater


@dataclass
class Config:
    """Options for running one bundled manipulator."""

    robot: Literal[ROBOT_NAMES]
    environment: Literal[ENVIRONMENT_NAMES] = "empty"
    controller: Literal[CONTROLLER_NAMES] = "none"
    headless: bool = False
    steps: int = 1000
    dt: float = 0.002


def main() -> None:
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    config = tyro.cli(Config, description="Run a manipulator in a MuJoCo scene")
    if config.steps < 0:
        logger.error("--steps must be zero or greater", exit_code=2)

    try:
        simulator = Simulator(
            create_environment(config.environment),
            robots=[RobotSpec(config.robot, config.robot)],
            dt=config.dt,
        )
        robot = simulator.robots[config.robot]
        robot.change_controller(create_controller(config.controller, robot))
        if config.robot == "forte" and config.controller in ("pd", "osc"):
            simulator.target_updater = demo_target_updater(
                simulator, {robot.name: config.controller}
            )
    except ValueError as error:
        logger.error(str(error), exit_code=2)
    data = simulator.data
    if config.headless:
        stats = simulator.run_steps(config.steps)[robot.name]
        print(
            f"{config.robot}: simulated {data.time:.3f} seconds; "
            f"environment = {config.environment}; qpos = {data.qpos}"
        )
        if robot.controller is not None:
            unit = {"pd": "joint units", "position": "joint units", "osc": "m"}[config.controller]
            print(stats.describe("tracking error", unit))
        return

    name = "manipulator"
    manager.add_simulator(name, simulator)
    try:
        manager.show(name)
    finally:
        if manager.simulators.get(name) is simulator:
            manager.remove_simulator(name)


if __name__ == "__main__":
    main()
