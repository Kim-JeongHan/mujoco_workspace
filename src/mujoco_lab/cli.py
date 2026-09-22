"""Command-line entry points for simulation, viewing, and rendering."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from mujoco_lab.assets import ENVIRONMENT_NAMES, ROBOT_NAMES
from mujoco_lab.control import CONTROLLER_NAMES, create_controller, demo_target_updater
from mujoco_lab.environment import create_environment
from mujoco_lab.robot import RobotSpec
from mujoco_lab.simulation import Simulator
from mujoco_lab.simulator_manager import SimulatorManager

Command = Literal["simulate", "view", "render"]
RobotName = Literal[ROBOT_NAMES]
EnvironmentName = Literal[ENVIRONMENT_NAMES]
ControllerName = Literal[CONTROLLER_NAMES]


@dataclass
class Config:
    """Options shared by the simulation, viewer, and renderer commands."""

    command: Command  # simulate, view, or render
    steps: int = 100  # physics steps
    output: Path = Path("outputs/scene.png")  # rendered PNG path
    dt: float = 0.002  # physics and control period in seconds
    controller: ControllerName = "none"  # none, position, pd, or osc
    environment: EnvironmentName | None = None  # optional scene
    robot: RobotName = "forte"  # robot asset


def main() -> None:
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    config = tyro.cli(Config, description="MuJoCo workspace")
    steps = config.steps
    if steps is None:
        steps = 0 if config.command == "render" else 1000
    if steps < 0:
        logger.error("--steps must be zero or greater", exit_code=2)
    if not (config.robot or config.environment):
        logger.error("select --robot or --environment", exit_code=2)
    if config.controller != "none" and not config.robot:
        logger.error(
            "--controller requires --robot",
            exit_code=2,
        )
    try:
        if config.robot:
            simulator = Simulator(
                create_environment(config.environment or "empty"),
                robots=[RobotSpec(config.robot, config.robot)],
                dt=config.dt,
            )
        else:
            simulator = Simulator(
                create_environment(config.environment),
                dt=config.dt,
            )
        robot = next(iter(simulator.robots.values()), None)
        if robot is not None:
            robot.change_controller(create_controller(config.controller, robot))
            if config.robot == "forte" and config.controller in ("pd", "osc"):
                simulator.target_updater = demo_target_updater(
                    simulator, {robot.name: config.controller}
                )
    except ValueError as error:
        logger.error(str(error), exit_code=2)

    if config.command == "view":
        name = "cli"
        manager.add_simulator(name, simulator)
        try:
            manager.show(name)
        except KeyboardInterrupt:
            pass
        finally:
            if manager.simulators.get(name) is simulator:
                manager.remove_simulator(name)
        return

    simulator.run_steps(steps)
    if config.command == "render":
        name = "cli"
        manager.add_simulator(name, simulator)
        try:
            logger.info(str(manager.save_frame(name, config.output)))
        finally:
            if manager.simulators.get(name) is simulator:
                manager.remove_simulator(name)
        return
