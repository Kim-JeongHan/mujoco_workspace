"""Observation metadata follows each environment's actual robot layout."""

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.learning.envs import CubeStackEnv
from mujoco_lab.learning.envs.cube_stack import cube_stack_observation_layout
from mujoco_lab.utils import Transform


@pytest.mark.parametrize("robots", [1, 2])
def test_cube_metadata_identifies_all_grasp_and_cube_rotations(robots):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[
            RobotSpec(
                f"robot{index}",
                "forte",
                Transform(translation=[index * 0.8, 0, 0]),
                config=load_robot_config("forte"),
            )
            for index in range(robots)
        ],
    )
    env = CubeStackEnv(CubeStackTask(simulator, 2), xy_range=0)
    obs, _ = env.reset(seed=0)
    metadata = env.observation_metadata
    assert metadata["frame_dim"] == len(obs) == 24 * robots + 2 * (12 + 3 * robots)
    matrices = [simulator.data.site_xmat[env._grasp_sites[robot.name]] for robot in env.robots]
    matrices.extend(simulator.data.xmat[body] for body in env._cube_bodies)
    expected = np.concatenate([matrix.reshape(3, 3)[:, :2].reshape(-1) for matrix in matrices])
    np.testing.assert_allclose(obs[metadata["rotation_indices"]], expected, atol=1e-7)
    if robots == 1:
        width, rotations = cube_stack_observation_layout(2)
        assert metadata == {"frame_dim": width, "rotation_indices": rotations}
    assert np.isfinite(env.camera_lookat).all()
