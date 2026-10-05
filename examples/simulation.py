"""Run a scene with an optional robot: uv run python examples/simulation.py --robot panda."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import tyro

from mujoco_lab import ENVIRONMENT_NAMES, RobotSpec, Simulator, SimulatorManager, create_environment
from mujoco_lab.assets import RobotName
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import create_controller


@dataclass
class Config:
    """Choose a registered environment or MJCF/URDF file and an optional robot."""

    environment: str = "empty"
    robot: RobotName | None = None
    headless: bool = False
    steps: int = 1000
    dt: float = 0.002


def main() -> None:
    config = tyro.cli(Config, description="Simulate an environment with an optional robot")
    manager = SimulatorManager()
    if config.steps < 0:
        manager.logger.error("--steps must be zero or greater", exit_code=2)

    if config.environment in ENVIRONMENT_NAMES:
        scene = create_environment(config.environment)
    else:
        scene = mujoco.MjSpec.from_file(str(Path(config.environment).expanduser().resolve()))
    robots = []
    if config.robot is not None:
        robots.append(RobotSpec(config.robot, config.robot, config=load_robot_config(config.robot)))
    simulator = Simulator(scene, robots=robots, dt=config.dt)
    for robot in simulator.robots.values():
        robot.change_controller(create_controller(robot, robot.config.controller))

    if config.headless:
        simulator.run_steps(config.steps)
        manager.logger.info(
            f"environment={config.environment} robot={config.robot or 'none'} "
            f"simulated={simulator.data.time:.3f}s"
        )
    else:
        manager.add_simulator("simulation", simulator)
        manager.show("simulation")


if __name__ == "__main__":
    main()
