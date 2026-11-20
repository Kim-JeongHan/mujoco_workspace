"""Build and evaluate book-insertion checkpoint environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import tyro

from mujoco_lab.learning.config import BCEvalConfig, EvaluationConfig
from mujoco_lab.learning.config.replay import BookReplayConfig
from mujoco_lab.learning.envs.book import BookEnv
from mujoco_lab.learning.envs.factory import create_env
from mujoco_lab.learning.metadata import EnvironmentMetadata


@dataclass
class BookEvalConfig(BCEvalConfig):
    """Allow enough simulated time for a complete book-insertion episode."""

    evaluation: EvaluationConfig = field(
        default_factory=lambda: EvaluationConfig(max_seconds=180.0)
    )


def create_book_evaluation_env(
    environment: EnvironmentMetadata,
    *,
    xy_range: float | None = None,
    max_steps: int,
    book_yaw_range_degrees: float | None = None,
) -> tuple[BookEnv, dict[str, Any]]:
    """Restore the recorded book, robot, controller tuning, and action cadence."""
    scene = environment.scene
    if not isinstance(scene, BookReplayConfig):
        raise ValueError("Expected a book-insertion environment")
    effective = replace(
        scene,
        xy_range=scene.xy_range if xy_range is None else xy_range,
        book_yaw_range_degrees=(
            scene.book_yaw_range_degrees
            if book_yaw_range_degrees is None
            else book_yaw_range_degrees
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
    """Evaluate a checkpoint recorded from book insertion."""
    from mujoco_lab.learning.evaluate import run
    from mujoco_lab.utils.logger import Logger

    config = tyro.cli(BookEvalConfig, description="Evaluate a saved book-insertion BC policy")
    run_dir, summary = run(config, expected_scene="book_insertion")
    Logger().info(f"Saved {summary['attempted']} evaluation attempts to {run_dir}")


if __name__ == "__main__":
    main()
