"""Run two robots in one scene: uv run python examples/multi_robot.py --headless."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_environment
from mujoco_lab.control import create_controller, demo_target_updater
from mujoco_lab.utils import Transform


@dataclass
class Config:
    """Options for running two robots in one shared MuJoCo scene."""

    layout: Literal["dual_forte", "forte_panda"] = "dual_forte"
    headless: bool = False
    steps: int = 1000
    dt: float = 0.002
    output: Path | None = None


def main() -> None:
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    config = tyro.cli(Config, description="Run two robots in one shared MuJoCo scene")
    if config.steps < 0:
        logger.error("--steps must be zero or greater", exit_code=2)
    right_asset = "forte" if config.layout == "dual_forte" else "panda"
    try:
        simulator = Simulator(
            create_environment("empty"),
            robots=[
                RobotSpec("left", "forte", Transform(translation=[-0.8, 0, 0])),
                RobotSpec("right", right_asset, Transform(translation=[0.8, 0, 0])),
            ],
            dt=config.dt,
        )
    except ValueError as error:
        logger.error(str(error), exit_code=2)
    left, right = simulator.robots["left"], simulator.robots["right"]
    left.change_controller(create_controller("pd", left))
    if right_asset == "forte":
        right.change_controller(create_controller("osc", right))
    simulator.target_updater = demo_target_updater(
        simulator, {"left": "pd", **({"right": "osc"} if right_asset == "forte" else {})}
    )
    if config.headless or config.output:
        stats = simulator.run_steps(config.steps)
        for name, robot in simulator.robots.items():
            robot.update_state()
            asset = "forte" if name == "left" else right_asset
            print(f"{name} ({asset}): qpos = {robot.joint_state.qpos}")
            print(stats[name].describe("tracking error", "m" if name == "right" else "rad"))
        print(f"Shared simulated seconds: {simulator.data.time:.3f}")
        if config.output:
            name = "multi_robot"
            manager.add_simulator(name, simulator)
            try:
                print(manager.save_frame(name, config.output))
            finally:
                if manager.simulators.get(name) is simulator:
                    manager.remove_simulator(name)
    else:
        name = "multi_robot"
        manager.add_simulator(name, simulator)
        try:
            manager.show(name)
        finally:
            if manager.simulators.get(name) is simulator:
                manager.remove_simulator(name)


if __name__ == "__main__":
    main()
