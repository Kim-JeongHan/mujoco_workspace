"""Build and evaluate cube-stack checkpoint environments."""

from __future__ import annotations

from typing import Any

import tyro

from mujoco_lab.learning.config import BCEvalConfig
from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.envs.factory import create_env


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
    action_repeat = metadata["architecture"]["physics_steps_per_action"]
    robot_type = replay["robot"]
    robot_name = replay["robot_name"]
    cubes = replay["cubes"]
    dt = replay["dt"]
    yaw_range = (
        replay["cube_yaw_range_degrees"]
        if cube_yaw_range_degrees is None
        else cube_yaw_range_degrees
    )
    xy_range = replay["xy_range"] if xy_range is None else xy_range
    min_gap = replay["min_gap"] if min_gap is None else min_gap
    env = create_env(
        CubeStackReplayConfig(
            robot=robot_type,
            cubes=cubes,
            environment=replay["environment"],
            xy_range=xy_range,
            min_gap=min_gap,
            cube_yaw_range_degrees=yaw_range,
            physics_steps_per_action=action_repeat,
        ),
        max_steps=max_steps,
        robot_name=robot_name,
        dt=dt,
    )
    scene = {
        "robot": robot_type,
        "robot_name": robot_name,
        "cubes": cubes,
        "environment": replay["environment"],
        "dt": env.simulator.dt,
        "simulation_hz": 1.0 / env.simulator.dt,
        "action_execution_hz": 1.0 / (env.simulator.dt * action_repeat),
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

    config = tyro.cli(BCEvalConfig, description="Evaluate a saved cube-stack BC policy")
    run_dir, summary = run(config, expected_scene="cube_stack")
    Logger().info(f"Saved {summary['attempted']} evaluation attempts to {run_dir}")


if __name__ == "__main__":
    main()
