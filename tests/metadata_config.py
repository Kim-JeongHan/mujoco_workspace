"""Provide explicit environment descriptions for small checkpoint fixtures."""

from mujoco_lab.learning.config.bc import BCTrainConfig
from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.metadata import EnvironmentMetadata


def checkpoint_environment(config: BCTrainConfig) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        CubeStackReplayConfig(
            robot=config.robot,
            cubes=1,
            physics_steps_per_action=config.physics_steps_per_action,
        ),
        dt=config.simulation_dt,
        robot_name=config.robot,
    )
