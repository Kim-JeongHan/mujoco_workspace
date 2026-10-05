"""Build and evaluate cube-stack checkpoint environments."""

from __future__ import annotations

from typing import Any

import tyro

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.control import create_controller
from mujoco_lab.learning.config.config import EvalConfig
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv


def create_cube_evaluation_env(
    metadata: dict[str, Any],
    *,
    xy_range: float | None,
    min_gap: float | None,
    max_steps: int,
    cube_yaw_range_degrees: float | None = None,
) -> tuple[CubeStackEnv, dict[str, Any]]:
    """Build the cube-stack environment from recorded scene metadata."""
    replay = metadata["dataset_metadata"]["replay"]
    action_repeat = (
        metadata["architecture"]["physics_steps_per_action"]
        if "architecture" in metadata
        else replay["physics_steps_per_action"]
    )
    robot_type = replay["robot"]
    robot_name = replay["robot_name"]
    cubes = replay["cubes"]
    dt = replay["dt"]
    yaw_range = (
        replay["cube_yaw_range_degrees"]
        if cube_yaw_range_degrees is None
        else cube_yaw_range_degrees
    )
    robot_config = load_robot_config(robot_type)
    simulator = Simulator(
        create_cube_stack(cubes, environment=replay["environment"]),
        robots=[RobotSpec(robot_name, robot_type, config=robot_config)],
        dt=dt,
    )
    robot = simulator.robots[robot_name]
    controller = create_controller(robot, robot_config.controller)
    robot.change_controller(controller)
    task = CubeStackTask(simulator, cubes)
    xy_range = replay.get("xy_range", 0.02) if xy_range is None else xy_range
    min_gap = replay.get("min_gap", 0.01) if min_gap is None else min_gap
    env = CubeStackEnv(
        task,
        xy_range=xy_range,
        min_gap=min_gap,
        cube_yaw_range_degrees=yaw_range,
        max_steps=max_steps,
        physics_steps_per_action=action_repeat,
    )
    scene = {
        "robot": robot_type,
        "robot_name": robot_name,
        "cubes": cubes,
        "environment": replay["environment"],
        "dt": simulator.dt,
        "simulation_hz": 1.0 / simulator.dt,
        "action_execution_hz": 1.0 / (simulator.dt * action_repeat),
        "physics_steps_per_action": action_repeat,
        "cube_yaw_range_degrees": yaw_range,
        "xy_range": xy_range,
        "min_gap": min_gap,
        "action_dt": env.action_dt,
    }
    return env, scene


def main() -> None:
    """Evaluate a checkpoint recorded from cube stacking."""
    from mujoco_lab.learning.evaluate import run
    from mujoco_lab.utils.logger import Logger

    config = tyro.cli(EvalConfig, description="Evaluate a saved cube-stack BC policy")
    run_dir, summary = run(config, expected_scene="cube_stack")
    Logger().info(f"Saved {summary['attempted']} evaluation attempts to {run_dir}")


if __name__ == "__main__":
    main()
