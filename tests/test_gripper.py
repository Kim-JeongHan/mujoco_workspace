"""Robot-owned gripper control alongside selected joint controllers."""

from dataclasses import replace

import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import GripperConfig, PoseConfig


def test_forte_gripper_controls_without_arm_controller_and_resets_home():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
    )
    robot = sim.robots["robot"]
    gripper = robot.gripper
    assert gripper is not None
    assert gripper.get_target() == pytest.approx(0.074)
    assert gripper.get_width() == pytest.approx(0.074)
    limits = gripper.get_control_limits()
    np.testing.assert_allclose(limits, [0, 0.074])
    limits[:] = 0  # Caller edits must not change the gripper's stored bounds.
    assert not gripper.is_active()
    gripper.set_target(0.02)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.01)
    assert gripper.is_active()
    sim.reset()
    assert gripper.get_target() == pytest.approx(0.074)
    assert not gripper.is_active()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.037)
    with pytest.raises(ValueError, match="gripper width"):
        gripper.set_target(-0.038)
    with pytest.raises(ValueError, match="gripper width"):
        gripper.set_target(np.nan)


@pytest.mark.parametrize("robot_type", ["panda", "forte"])
def test_gripper_clamps_endpoint_roundoff_and_rejects_larger_errors(robot_type):
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", robot_type, config=load_robot_config(robot_type))],
    )
    gripper = sim.robots["robot"].gripper
    lower, upper = gripper.get_control_limits()
    roundoff = np.finfo(float).eps * max(abs(lower), abs(upper))

    for boundary, direction in ((lower, -1), (upper, 1)):
        gripper.set_target(boundary + direction * roundoff)
        assert gripper.get_target() == boundary
        sim.step()
        assert sim.data.ctrl[gripper.actuator_id] == boundary / 2 * gripper.gear
        with pytest.raises(ValueError, match="gripper width"):
            gripper.set_target(boundary + direction * 1e-10)
        assert gripper.get_target() == boundary


@pytest.mark.parametrize("robot_type", ["panda", "forte"])
def test_grasp_contact_groups_exclude_visuals_and_require_each_finger(robot_type):
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", robot_type, config=load_robot_config(robot_type))],
    )
    gripper = sim.robots["robot"].gripper
    assert gripper is not None
    left, right = gripper.finger_geom_ids
    assert not gripper.has_contact_on_all_fingers(set())
    assert not gripper.has_contact_on_all_fingers(set(left))
    assert not gripper.has_contact_on_all_fingers(set(right))
    for left_geom in left:
        for right_geom in right:
            assert gripper.has_contact_on_all_fingers({left_geom, right_geom})
    visuals = {
        geom
        for geom in range(sim.model.ngeom)
        if not sim.model.geom_contype[geom] and not sim.model.geom_conaffinity[geom]
    }
    assert not gripper.has_contact_on_all_fingers(visuals)


def test_custom_gripper_opt_in_reserves_middle_actuator_slot(tmp_path, monkeypatch):
    path = tmp_path / "mini.xml"
    path.write_text("""
        <mujoco model="mini">
          <worldbody><body name="base">
            <joint name="alpha" type="hinge"/>
            <geom type="sphere" size=".1" mass="1"/>
            <body name="hand" pos="0 0 .2">
              <joint name="beta" type="hinge"/>
              <geom type="sphere" size=".05" mass="1"/>
              <body name="finger" pos="0 0 .1">
                <joint name="grip" type="slide" axis="0 1 0"/>
                <geom type="sphere" size=".02" mass=".1"/>
              </body>
            </body>
          </body></worldbody>
          <actuator>
            <motor name="beta_drive" joint="beta" gear="-2"/>
            <position name="grip_drive" joint="grip" kp="20" gear="2"/>
            <motor name="alpha_drive" joint="alpha" gear="3"/>
          </actuator>
          <keyframe><key name="home" ctrl="0 .06 0"/></keyframe>
        </mujoco>
    """)
    (tmp_path / "robot.yaml").write_text("""pose:
  default: [0, 0, 0]
controller:
  name: pd
  pd_gains:
    beta:
      kp: 1
      kd: 0
    alpha:
      kp: 1
      kd: 0
""")
    monkeypatch.setitem(ROBOT_ASSETS, "mini", RobotAsset(path))
    scene = create_environment("empty")
    plain = Simulator(
        scene, robots=[RobotSpec("plain", "mini", config=load_robot_config("mini"))]
    ).robots["plain"]
    assert plain.gripper is None

    config = replace(
        load_robot_config("mini"),
        gripper=GripperConfig(actuator="grip_drive", joints=["grip"]),
        pose=PoseConfig([0, 0, 0.03]),
    )
    sim = Simulator(scene, robots=[RobotSpec("custom", "mini", config=config)])
    robot = sim.robots["custom"]
    gripper = robot.gripper
    assert gripper is not None and gripper.slot == 1
    assert gripper.get_target() == pytest.approx(0.03)
    grip_qpos = sim.model.joint("custom/grip").qposadr[0]
    initial_position = gripper.get_width()
    sim.data.qpos[grip_qpos] = 0.012
    assert gripper.get_width() == pytest.approx(0.012)
    sim.data.qpos[grip_qpos] = initial_position
    robot.change_controller(create_test_controller(robot, controller="pd"))
    assert robot.control_joint_names == ("beta", "alpha")
    assert robot.target.position.shape == (2,)
    gripper.set_target(0.04)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.08)
    sim.reset()
    assert gripper.get_target() == pytest.approx(0.03)
    assert not gripper.is_active()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.06)

    path.write_text(
        path.read_text()
        .replace("</actuator>", '<motor name="grip_extra" joint="grip"/></actuator>')
        .replace('ctrl="0 .06 0"', 'ctrl="0 .06 0 0"')
    )
    shared = Simulator(scene, robots=[RobotSpec("shared", "mini", config=config)])
    with pytest.raises(ValueError, match="also drive the configured gripper joint"):
        create_test_controller(shared.robots["shared"], controller="pd")
