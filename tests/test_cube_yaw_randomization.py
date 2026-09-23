"""Seeded tabletop yaw and replay settings for cube demonstrations."""

import gc

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.randomization import sample_cube_positions
from mujoco_lab.learning.datasets.replay import replay_cube_yaw_range_degrees
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.rollout.collector import _replay_metadata_for_env
from mujoco_lab.tasks import CubeStackTask


def test_seeded_cube_yaw_rotates_qpos_and_keeps_rotated_footprints_apart():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    env = CubeStackEnv(CubeStackTask(simulator, 2), cube_yaw_range_degrees=45)
    first_obs, first_info = env.reset(seed=11)
    first_qpos = simulator.data.qpos.copy()
    obs, info = env.reset(seed=11)
    np.testing.assert_array_equal(obs, first_obs)
    np.testing.assert_array_equal(simulator.data.qpos, first_qpos)
    assert info == first_info

    yaws = np.asarray(info["cube_yaws_degrees"])
    assert yaws.shape == (2,)
    assert np.all(np.abs(yaws) <= 45)
    assert not np.allclose(yaws, 0)
    extents = []
    for index, (adr, yaw) in enumerate(zip(env._cube_qpos, yaws, strict=True)):
        expected = np.array([np.cos(np.deg2rad(yaw / 2)), 0, 0,
                             np.sin(np.deg2rad(yaw / 2))])
        np.testing.assert_allclose(simulator.data.qpos[adr + 3 : adr + 7], expected, atol=1e-12)
        assert simulator.data.qpos[adr + 2] == pytest.approx(simulator.model.qpos0[adr + 2])
        matrix = simulator.data.body(f"cube{index}/object_0").xmat.reshape(3, 3)
        np.testing.assert_allclose(matrix[:2, :2],
                                   [[np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw))],
                                    [np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))]],
                                   atol=1e-12)
        half = env._cube_half_sizes[index]
        extents.append(np.abs(matrix[:2, :2]) @ half)
    centers = np.array([simulator.data.qpos[adr : adr + 2] for adr in env._cube_qpos])
    assert np.any(np.abs(centers[0] - centers[1]) >= extents[0] + extents[1] + env.min_gap)
    # The observed rotation columns retain the physical yaw in the flat observation.
    observed_rotation = simulator.data.body("cube0/object_0").xmat.reshape(3, 3)[:, :2]
    np.testing.assert_allclose(obs[27:33], observed_rotation.reshape(-1), atol=1e-6)
    assert _replay_metadata_for_env(env, {})["cube_yaw_range_degrees"] == 45.0
    with pytest.raises(ValueError, match="Replay cube yaw range"):
        _replay_metadata_for_env(env, {"cube_yaw_range_degrees": 0.0})
    # With fixed close centers, default axis-aligned boxes fit, but the
    # sampled rotated footprints require more clearance and must be rejected.
    distance = sum(env._cube_half_sizes[:, 0]) + env.min_gap + 0.001
    env._home_xy[:] = [[0.0, 0.0], [distance, 0.0]]
    env.xy_range = 0.0
    assert len(sample_cube_positions(
        env._home_xy, env._cube_half_sizes, np.random.default_rng(11), 0.0,
        env.min_gap, 100,
    )) == 2
    with pytest.raises(ValueError, match="without overlap"):
        env.reset(seed=11)


def test_zero_yaw_keeps_legacy_xy_draws_and_home_quaternion():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    env = CubeStackEnv(CubeStackTask(simulator, 2))
    expected_xy = sample_cube_positions(
        env._home_xy, env._cube_half_sizes, np.random.default_rng(11), env.xy_range,
        env.min_gap, 100,
    )
    obs, info = env.reset(seed=11)
    assert info["cube_yaws_degrees"] == [0.0, 0.0]
    for adr, xy, home in zip(env._cube_qpos, expected_xy, env._home_quat, strict=True):
        np.testing.assert_array_equal(simulator.data.qpos[adr : adr + 2], xy)
        np.testing.assert_array_equal(simulator.data.qpos[adr + 3 : adr + 7], home)
    assert np.isfinite(obs).all()
    with pytest.raises(ValueError, match="cube_yaw_range_degrees"):
        CubeStackEnv(CubeStackTask(simulator, 2), cube_yaw_range_degrees=float("nan"))


def test_evaluation_restores_recorded_yaw(monkeypatch):
    monkeypatch.setattr(
        "mujoco_lab.learning.evaluate._verify_replay", lambda *_args, **_kwargs: None
    )
    metadata = {
        "dataset_metadata": {"replay": {
            "robot": "forte", "cubes": 2, "environment": "table_shelf", "dt": 0.002,
            "cube_yaw_range_degrees": 45.0,
        }}
    }
    env, _, scene = create_evaluation_env(metadata, xy_range=0.02, min_gap=0.01, max_steps=1)
    assert env.cube_yaw_range_degrees == scene["cube_yaw_range_degrees"] == 45.0
    assert max(map(abs, env.reset(seed=11)[1]["cube_yaws_degrees"])) > 0
    del env
    gc.collect()
    overridden, _, scene = create_evaluation_env(
        metadata, xy_range=0.02, min_gap=0.01, max_steps=1,
        cube_yaw_range_degrees=0.0,
    )
    assert overridden.cube_yaw_range_degrees == scene["cube_yaw_range_degrees"] == 0.0
    del overridden
    gc.collect()
    del metadata["dataset_metadata"]["replay"]["cube_yaw_range_degrees"]
    legacy, _, scene = create_evaluation_env(
        metadata, xy_range=0.02, min_gap=0.01, max_steps=1,
    )
    assert legacy.cube_yaw_range_degrees == scene["cube_yaw_range_degrees"] == 0.0
    assert replay_cube_yaw_range_degrees({}) == 0.0
    assert replay_cube_yaw_range_degrees({"cube_yaw_range_degrees": 0}) == 0.0
    for value in (-1, float("inf"), True):
        with pytest.raises(ValueError, match="cube_yaw_range_degrees"):
            replay_cube_yaw_range_degrees({"cube_yaw_range_degrees": value})
