"""Build and evaluate cube-stack checkpoint environments."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import tyro

from mujoco_lab.learning.config import BCEvalConfig
from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.envs.factory import create_env
from mujoco_lab.learning.metadata import EnvironmentMetadata


def create_cube_evaluation_env(
    environment: EnvironmentMetadata,
    *,
    xy_range: float | None,
    min_gap: float | None,
    max_steps: int,
    cube_yaw_range_degrees: float | None = None,
) -> tuple[CubeStackEnv, dict[str, Any]]:
    """Build the cube-stack environment from recorded scene metadata."""
    scene = environment.scene
    if not isinstance(scene, CubeStackReplayConfig):
        raise ValueError("Expected a cube-stack environment")
    effective = replace(
        scene,
        xy_range=scene.xy_range if xy_range is None else xy_range,
        min_gap=scene.min_gap if min_gap is None else min_gap,
        cube_yaw_range_degrees=(
            scene.cube_yaw_range_degrees
            if cube_yaw_range_degrees is None
            else cube_yaw_range_degrees
        ),
    )
    env = create_env(
        effective,
        max_steps=max_steps,
        robot_name=environment.robot_name,
        dt=environment.dt,
    )
    return env, {
        **asdict(effective),
        "robot_name": environment.robot_name,
        "dt": env.simulator.dt,
        "simulation_hz": 1.0 / env.simulator.dt,
        "action_execution_hz": 1.0 / env.action_dt,
        "action_dt": env.action_dt,
    }


def main() -> None:
    """Evaluate a checkpoint recorded from cube stacking."""
    from mujoco_lab.learning.evaluate import run
    from mujoco_lab.utils.logger import Logger

    config = tyro.cli(BCEvalConfig, description="Evaluate a saved cube-stack BC policy")
    run_dir, summary = run(config, expected_scene="cube_stack")
    Logger().info(f"Saved {summary['attempted']} evaluation attempts to {run_dir}")


if __name__ == "__main__":
    main()
