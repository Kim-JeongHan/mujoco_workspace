"""Load an external model: uv run python examples/model.py --model scene.xml."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import tyro

from mujoco_lab import Simulator, SimulatorManager


@dataclass
class Config:
    """Options for loading and running an MJCF or URDF file."""

    model: Path
    headless: bool = False
    steps: int = 1000
    dt: float = 0.002


def main() -> None:
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    config = tyro.cli(Config, description="Load an external MuJoCo model")
    if config.steps < 0:
        logger.error("--steps must be zero or greater", exit_code=2)

    try:
        scene = mujoco.MjSpec.from_file(str(config.model.expanduser().resolve()))
        simulator = Simulator(
            scene,
            dt=config.dt,
        )
    except ValueError as error:
        logger.error(str(error), exit_code=2)

    if config.headless:
        simulator.run_steps(config.steps)
        logger.info(
            f"{config.model}: simulated {simulator.data.time:.3f} seconds; "
            f"qpos = {simulator.data.qpos}"
        )
        return

    name = "model"
    manager.add_simulator(name, simulator)
    try:
        manager.show(name)
    finally:
        if manager.simulators.get(name) is simulator:
            manager.remove_simulator(name)


if __name__ == "__main__":
    main()
