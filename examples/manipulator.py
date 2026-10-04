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
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import (
    JointSpacePD,
    OperationalSpaceControl,
    create_controller,
    demo_target_updater,
)


@dataclass
class Config:
    """Options for running one bundled manipulator."""

    robot: Literal[ROBOT_NAMES]
    environment: Literal[ENVIRONMENT_NAMES] = "empty"
    headless: bool = False
    steps: int = 1000
    dt: float = 0.002


def main() -> None:
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    config = tyro.cli(Config, description="Run a manipulator in a MuJoCo scene")
    if config.steps < 0:
        logger.error("--steps must be zero or greater", exit_code=2)

    robot_config = load_robot_config(config.robot)
    simulator = Simulator(
        create_environment(config.environment),
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
        dt=config.dt,
    )
    robot = simulator.robots[config.robot]
    controller = create_controller(robot, robot_config.controller)
    robot.change_controller(controller)
    if isinstance(controller, (JointSpacePD, OperationalSpaceControl)):
        controller_name = "pd" if isinstance(controller, JointSpacePD) else "osc"
        simulator.target_updater = demo_target_updater(simulator, {robot.name: controller_name})
    data = simulator.data
    if config.headless:
        stats = simulator.run_steps(config.steps)[robot.name]
        logger.info(
            f"{config.robot}: simulated {data.time:.3f} seconds; "
            f"environment = {config.environment}; qpos = {data.qpos}"
        )
        if robot.controller is not None:
            unit = "m" if isinstance(robot.controller, OperationalSpaceControl) else "joint units"
            logger.info(stats.describe("tracking error", unit))
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
