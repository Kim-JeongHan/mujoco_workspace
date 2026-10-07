"""Cube spawn randomization preserves a usable MuJoCo scene."""

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config, randomize_cube_positions
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv


def _cube_positions(scene):
    return np.array([scene.body(f"cube{i}/object_0").pos for i in range(2)])


def test_randomization_is_seeded_and_compiles_with_forte_task():
    original = create_cube_stack(2)
    scene = create_cube_stack(2)
    same_seed = create_cube_stack(2)
    starts = _cube_positions(original)
    targets = [np.array(scene.body(f"cube{i}/object_target_0").pos) for i in range(2)]
    orientations = [np.array(scene.body(f"cube{i}/object_0").quat) for i in range(2)]

    randomize_cube_positions(scene, seed=12)
    randomize_cube_positions(same_seed, seed=12)
    positions = _cube_positions(scene)
    np.testing.assert_array_equal(positions, _cube_positions(same_seed))
    assert np.any(positions[:, :2] != starts[:, :2])
    assert np.all(np.abs(positions[:, :2] - starts[:, :2]) <= 0.02)
    np.testing.assert_array_equal(positions[:, 2], starts[:, 2])
    for i in range(2):
        np.testing.assert_array_equal(scene.body(f"cube{i}/object_0").quat, orientations[i])
        np.testing.assert_array_equal(scene.body(f"cube{i}/object_target_0").pos, targets[i])

    simulator = Simulator(
        scene, robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))]
    )
    task = CubeStackTask(simulator, 2)
    for i in range(2):
        world_start = simulator.data.body(f"cube{i}/object_0").xpos
        np.testing.assert_allclose(world_start, positions[i] + [0, 0, 0.8])
        np.testing.assert_allclose(task.starts[i], world_start)
        np.testing.assert_allclose(task.goals[i], targets[i] + [0, 0, 0.8])
    simulator.reset()
    for i in range(2):
        np.testing.assert_allclose(simulator.data.body(f"cube{i}/object_0").xpos, task.starts[i])


@pytest.mark.parametrize("yaw_range_degrees", [0.0, 45.0])
def test_impossible_layout_does_not_move_any_cube_or_target(yaw_range_degrees):
    scene = create_cube_stack(2)
    first = np.array(scene.body("cube0/object_0").pos)
    # Centers are 3 cm apart: separated points, but overlapping 4 cm cubes.
    scene.body("cube1/object_0").pos = first + [0.03, 0, 0]
    before = _cube_positions(scene)
    targets = [np.array(scene.body(f"cube{i}/object_target_0").pos) for i in range(2)]
    orientations = [scene.body(f"cube{i}/object_0").quat.copy() for i in range(2)]

    with pytest.raises(ValueError, match="Could not place cube"):
        randomize_cube_positions(
            scene, xy_range=0.005, min_gap=0.01, yaw_range_degrees=yaw_range_degrees, seed=2
        )
    np.testing.assert_array_equal(_cube_positions(scene), before)
    for i in range(2):
        np.testing.assert_array_equal(scene.body(f"cube{i}/object_target_0").pos, targets[i])
        np.testing.assert_array_equal(scene.body(f"cube{i}/object_0").quat, orientations[i])


@pytest.mark.parametrize("yaw_range_degrees", [0.0, 45.0])
def test_scene_randomization_matches_environment_reset_and_preserves_reset_pose(yaw_range_degrees):
    robot_config = load_robot_config("forte")
    baseline = Simulator(
        create_cube_stack(2), robots=[RobotSpec("forte", "forte", config=robot_config)]
    )
    env = CubeStackEnv(CubeStackTask(baseline, 2), cube_yaw_range_degrees=yaw_range_degrees)
    _, info = env.reset(seed=11)
    scene = create_cube_stack(2)
    randomize_cube_positions(scene, yaw_range_degrees=yaw_range_degrees, seed=11)
    simulator = Simulator(scene, robots=[RobotSpec("forte", "forte", config=robot_config)])
    for index, yaw in enumerate(info["cube_yaws_degrees"]):
        body = f"cube{index}/object_0"
        np.testing.assert_allclose(simulator.data.body(body).xpos, baseline.data.body(body).xpos)
        np.testing.assert_allclose(simulator.data.body(body).xquat, baseline.data.body(body).xquat)
        expected = [np.cos(np.deg2rad(yaw / 2)), 0.0, 0.0, np.sin(np.deg2rad(yaw / 2))]
        np.testing.assert_allclose(simulator.data.body(body).xquat, expected, atol=1e-12)
    initial_qpos = simulator.data.qpos.copy()
    simulator.run_steps(1)
    simulator.reset()
    np.testing.assert_array_equal(simulator.data.qpos, initial_qpos)
