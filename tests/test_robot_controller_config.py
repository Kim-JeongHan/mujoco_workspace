"""Robot-local YAML defaults, joint gains, and task-specific tuning."""

from dataclasses import replace

import numpy as np
import pytest
import yaml
from dacite import UnexpectedDataError

from mujoco_lab import RobotSpec, Simulator, create_cube_stack, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import (
    ControllerConfig,
    PoseConfig,
    RobotConfig,
)
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.control import create_controller
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv


def test_yaml_position_limits_use_degrees_without_changing_motion_limits(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pose": {"default": [0, 0]},
                "controller": {"name": "none"},
                "constraints": {
                    "joint_names": ["joint1", "joint2"],
                    "position_limit": [[-180, 90], [-45, 0]],
                    "velocity_limit": [1.0, 2.0],
                    "acceleration_limit": [3.0, 4.0],
                },
            }
        )
    )
    for _ in range(2):
        config = RobotConfig.load(path)
        np.testing.assert_allclose(
            config.constraints.position_limit, [[-np.pi, np.pi / 2], [-np.pi / 4, 0]]
        )
        assert config.constraints.velocity_limit == [1.0, 2.0]
        assert config.constraints.acceleration_limit == [3.0, 4.0]
    assert yaml.safe_load(path.read_text())["constraints"]["position_limit"] == [
        [-180, 90],
        [-45, 0],
    ]


@pytest.mark.parametrize("with_gripper", [False, True])
@pytest.mark.parametrize("with_zero", [False, True])
def test_yaml_pose_preserves_gripper_position_and_converts_arm_angles(
    tmp_path, with_gripper, with_zero
):
    default = [-90, 90]
    zero = [180, -45]
    settings = {"controller": {"name": "none"}, "pose": {"default": default}}
    if with_zero:
        settings["pose"]["zero"] = zero
    if with_gripper:
        default.append(-0.02)
        zero.append(0.03)
        settings["gripper"] = {"actuator": "grip_drive", "joints": ["grip"]}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(settings))
    config = RobotConfig.load(path)
    assert isinstance(config.pose, PoseConfig)
    expected = [-np.pi / 2, np.pi / 2]
    if with_gripper:
        expected.append(-0.02)
    np.testing.assert_allclose(config.pose.default, expected)
    if with_zero:
        expected_zero = [np.pi, -np.pi / 4]
        if with_gripper:
            expected_zero.append(0.03)
        np.testing.assert_allclose(config.pose.zero, expected_zero)
        assert config.pose.named_poses() == {"default": expected, "zero": expected_zero}
    else:
        assert config.pose.zero is None
        assert config.pose.named_poses() == {"default": expected}
    assert yaml.safe_load(path.read_text())["pose"]["default"] == default
    if with_zero:
        assert yaml.safe_load(path.read_text())["pose"]["zero"] == zero


def test_yaml_pose_rejects_unsupported_pose_names(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "controller": {"name": "none"},
                "pose": {"default": [0], "ready": [90]},
            }
        )
    )
    with pytest.raises(UnexpectedDataError, match="ready"):
        RobotConfig.load(path)


@pytest.mark.parametrize("robot_type", ["forte", "panda"])
def test_default_pose_initializes_joints_controls_and_environment_reset(robot_type):
    config = load_robot_config(robot_type)
    pose = config.pose.default.copy()
    pose[0], pose[-1] = 0.12, 0.02
    config = replace(config, pose=PoseConfig(pose))
    sim = Simulator(create_cube_stack(1), robots=[RobotSpec("arm", robot_type, config=config)])
    robot = sim.robots["arm"]
    assert robot.gripper is not None
    _, slots = robot.get_arm_joint_mapping()
    np.testing.assert_allclose(robot.state.snapshot().qpos[slots], pose[:-1])
    finger_position = sim.model.jnt_range[robot.gripper.joint_id, 0] + pose[-1] / 2
    np.testing.assert_allclose(sim.data.qpos[robot.gripper._finger_qpos_indices], finger_position)
    assert robot.gripper.get_width() == pytest.approx(pose[-1])
    assert robot.gripper.get_target() == pytest.approx(pose[-1])
    assert sim.data.ctrl[robot.gripper.actuator_id] == pytest.approx(
        finger_position * robot.gripper.gear
    )
    robot.change_controller(create_controller(robot, config.controller))
    assert robot.target is not None
    np.testing.assert_allclose(robot.target.position, pose[:-1])
    expected = {name: getattr(sim.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
    sim.data.qpos[robot.state.qpos_indices] += 0.2
    sim.data.qvel[robot.state.dof_indices] = 0.3
    sim.data.ctrl[robot.actuator_ids] = 1
    robot.gripper.set_target(0)
    env = CubeStackEnv(CubeStackTask(sim, 1), xy_range=0)
    env.reset(seed=0)
    for name, values in expected.items():
        np.testing.assert_allclose(getattr(sim.data, name), values)
    np.testing.assert_allclose(robot.target.position, pose[:-1])
    assert robot.gripper.get_target() == pytest.approx(pose[-1])
    assert not robot.gripper.is_active()


def test_shared_config_gets_independent_asset_metadata(tmp_path, monkeypatch):
    from mujoco_lab.utils import Transform

    for name in ("a", "b"):
        path = tmp_path / f"{name}.xml"
        path.write_text(f"""<mujoco>
          <worldbody><body name="root_{name}">
            <joint name="joint_{name}"/><geom size=".1"/><site name="tip_{name}"/>
          </body></worldbody>
        </mujoco>""")
        monkeypatch.setitem(ROBOT_ASSETS, name, RobotAsset(path))
    config = RobotConfig(ControllerConfig(name="none"), pose=PoseConfig([0]))
    initial_info = config.model_info
    sim = Simulator(
        create_environment("empty"),
        robots=[
            RobotSpec("left", "a", Transform(translation=[-0.5, 0, 1]), config=config),
            RobotSpec("right", "b", Transform(translation=[0.5, 0, 1]), config=config),
        ],
    )
    for name, suffix in (("left", "a"), ("right", "b")):
        robot = sim.robots[name]
        assert robot.config.model_info.root_name == f"root_{suffix}"
        assert robot.config.model_info.joint_names == (f"joint_{suffix}",)
        assert robot.state.site_id(f"tip_{suffix}") == sim.model.site(f"{name}/tip_{suffix}").id
    assert sim.robots["left"].config.model_info is not sim.robots["right"].config.model_info
    assert config.model_info is initial_info
