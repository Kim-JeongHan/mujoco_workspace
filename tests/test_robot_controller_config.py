"""Robot-local YAML defaults, joint gains, and task-specific tuning."""

from dataclasses import replace

import numpy as np
import pytest
import yaml
from dacite import WrongTypeError

from mujoco_lab import RobotSpec, Simulator, create_cube_stack, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import (
    Constraints,
    ControllerConfig,
    PoseConfig,
    RobotConfig,
)
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.control import (
    JointSpaceComputedTorque,
    PositionController,
    create_controller,
)
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv


@pytest.mark.parametrize(
    ("robot_type", "mode", "algorithm"),
    [
        ("forte", "computed_torque", JointSpaceComputedTorque),
        ("panda", "position", PositionController),
    ],
)
def test_robot_creates_configured_controller_and_target_at_initial_pose(
    robot_type, mode, algorithm
):
    config = load_robot_config(robot_type)
    config.controller.name = mode
    simulator = Simulator(
        create_environment("empty"), robots=[RobotSpec("robot", robot_type, config=config)]
    )
    robot = simulator.robots["robot"]
    assert isinstance(robot.controller, algorithm)
    np.testing.assert_array_equal(robot.target.position, robot.get_control_state().qpos)
    np.testing.assert_array_equal(robot.reference_generator.state().position, robot.target.position)
    np.testing.assert_array_equal(robot.target.velocity, 0)
    np.testing.assert_array_equal(robot.target.acceleration, 0)


def test_yaml_rejects_controller_none(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text("controller:\n  name: none\npose:\n  default: [0]\n")
    with pytest.raises(WrongTypeError, match="controller.name"):
        RobotConfig.load(path)


def test_controller_replacement_resets_target_and_reference():
    simulator = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
    )
    robot = simulator.robots["robot"]
    reference = robot.reference_generator
    replacement = create_controller(
        robot, ControllerConfig("computed_torque", natural_frequency=20)
    )
    robot.change_controller(replacement)
    assert robot.controller is replacement
    assert robot.reference_generator is not reference
    np.testing.assert_array_equal(robot.target.position, robot.get_control_state().qpos)


def test_robotless_simulator_can_run_native_actuators():
    import mujoco

    scene = create_environment("empty")
    body = scene.worldbody.add_body(name="native", pos=[0, 0, 1])
    body.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=1)
    scene.add_actuator(name="native_drive", target="hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    simulator = Simulator(scene)
    simulator.data.ctrl[:] = 0.3
    assert simulator.run_steps(3) is None
    assert not simulator.robots
    np.testing.assert_array_equal(simulator.data.ctrl, [0.3])
    assert simulator.data.qvel[0] > 0
    assert simulator.data.time == pytest.approx(3 * simulator.dt)


def test_yaml_position_limits_use_degrees_without_changing_motion_limits(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pose": {"default": [0, 0]},
                "controller": {"name": "position"},
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
        assert config.constraints is not None
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
    settings = {
        "controller": {"name": "position"},
        "pose": {"default": default},
        "constraints": {
            "joint_names": ["joint1", "joint2"],
            "position_limit": [[-180, 180], [-180, 180]],
            "velocity_limit": [1.0, 1.0],
            "acceleration_limit": [2.0, 2.0],
        },
    }
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
        assert config.pose.zero is not None
        np.testing.assert_allclose(config.pose.zero, expected_zero)
        assert config.pose.named_poses() == {"default": expected, "zero": expected_zero}
    else:
        assert config.pose.zero is None
        assert config.pose.named_poses() == {"default": expected}
    assert yaml.safe_load(path.read_text())["pose"]["default"] == default
    if with_zero:
        assert yaml.safe_load(path.read_text())["pose"]["zero"] == zero


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
          <actuator><motor name="drive_{name}" joint="joint_{name}"/></actuator>
        </mujoco>""")
        monkeypatch.setitem(ROBOT_ASSETS, name, RobotAsset(path))
    config = RobotConfig(
        ControllerConfig(name="computed_torque"),
        Constraints(["joint_a", "joint_b"], [[-np.pi, np.pi]] * 2, [1.0] * 2, [2.0] * 2),
        pose=PoseConfig([0]),
    )
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
