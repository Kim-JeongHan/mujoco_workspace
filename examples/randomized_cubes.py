"""View randomized cube placements: uv run python examples/randomized_cubes.py."""

from dataclasses import dataclass

import tyro

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_cube_stack
from mujoco_lab.assets.loader import randomize_cube_positions
from mujoco_lab.control import create_controller


@dataclass
class Config:
    """Options for viewing randomized cubes beside a stationary Forte arm."""

    cubes: int = 2
    seed: int = 42
    xy_range: float = 0.02
    min_gap: float = 0.01
    headless: bool = False
    steps: int = 1000


def main() -> None:
    """Build and display a fresh randomized cube scene with Forte at home."""
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    config = tyro.cli(Config, description="View randomized cubes beside Forte")
    if config.steps < 0:
        logger.error("--steps must be zero or greater", exit_code=2)

    try:
        scene = create_cube_stack(config.cubes)
        randomize_cube_positions(
            scene,
            xy_range=config.xy_range,
            min_gap=config.min_gap,
            seed=config.seed,
        )
        simulator = Simulator(scene, robots=[RobotSpec("forte", "forte")])
        robot = simulator.robots["forte"]
        robot.change_controller(create_controller("pd", robot))
        robot.gripper.set_target(0.0)
    except ValueError as error:
        logger.error(str(error), exit_code=2)

    name = "randomized_cubes"
    if config.headless:
        stats = simulator.run_steps(config.steps)[robot.name]
        cube_xy = [
            simulator.data.body(f"cube{index}/object_0").xpos[:2].round(3).tolist()
            for index in range(config.cubes)
        ]
        logger.info(f"simulated {simulator.data.time:.3f} seconds; cube XY positions = {cube_xy}")
        logger.info(stats.describe("tracking error", "rad"))
        return

    manager.add_simulator(name, simulator)
    try:
        manager.show(name)
    finally:
        if manager.simulators.get(name) is simulator:
            manager.remove_simulator(name)


if __name__ == "__main__":
    main()
