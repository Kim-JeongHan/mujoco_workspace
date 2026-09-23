"""Robot-owned gripper control alongside selected joint controllers."""

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.control import create_controller


def test_forte_gripper_controls_without_arm_controller_and_resets_home():
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("robot", "forte")])
    robot = sim.robots["robot"]
    gripper = robot.gripper
    assert gripper is not None
    assert gripper.get_target() == pytest.approx(0)
    assert gripper.get_position() == pytest.approx(
        sim.data.qpos[sim.model.joint("robot/gripper_left_joint").qposadr[0]]
    )
    assert not gripper.is_active()
    gripper.set_target(-0.01)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(-0.01)
    assert gripper.is_active()
    sim.reset()
    assert gripper.get_target() == pytest.approx(0)
    assert not gripper.is_active()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0)
    with pytest.raises(ValueError, match="gripper target"):
        gripper.set_target(-0.025)
    with pytest.raises(ValueError, match="gripper target"):
        gripper.set_target(np.nan)


def test_panda_arm_position_target_excludes_gripper_and_preserves_gripper_override():
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("robot", "panda")])
    robot = sim.robots["robot"]
    gripper = robot.gripper
    assert gripper is not None
    robot.change_controller(create_controller("position", robot))
    assert robot.target.position.shape == (7,)
    assert len(robot.control_joint_names) == 7
    gripper.set_target(0.02)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.02)
    robot.change_controller(None)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.02)
    robot.change_controller(create_controller("position", robot))
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.02)
    sim.reset()
    assert not gripper.is_active()
    home_ctrl = sim.data.ctrl[gripper.actuator_id]
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(home_ctrl)


def test_registered_gripper_default_and_instance_override(tmp_path, monkeypatch):
    path = tmp_path / "two_grippers.xml"
    path.write_text("""
        <mujoco model="two_grippers">
          <worldbody><body name="base">
            <joint name="left" type="slide"/>
            <geom type="sphere" size=".1" mass="1"/>
            <body name="right_body">
              <joint name="right" type="slide"/>
              <geom type="sphere" size=".05" mass="1"/>
            </body>
          </body></worldbody>
          <actuator>
            <position name="left_drive" joint="left" kp="20"/>
            <position name="right_drive" joint="right" kp="20"/>
          </actuator>
        </mujoco>
    """)
    monkeypatch.setitem(
        ROBOT_ASSETS, "two_grippers", RobotAsset(path, gripper_actuator="left_drive")
    )
    scene = create_environment("empty")
    default = Simulator(scene, robots=[RobotSpec("robot", "two_grippers")]).robots["robot"]
    override = Simulator(
        scene, robots=[RobotSpec("robot", "two_grippers", gripper_actuator="right_drive")]
    ).robots["robot"]
    assert default.gripper is not None and default.gripper.slot == 0
    assert override.gripper is not None and override.gripper.slot == 1


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
    monkeypatch.setitem(ROBOT_ASSETS, "mini", RobotAsset(path))
    scene = create_environment("empty")
    plain = Simulator(scene, robots=[RobotSpec("plain", "mini")]).robots["plain"]
    assert plain.gripper is None

    sim = Simulator(scene, robots=[RobotSpec("custom", "mini", gripper_actuator="grip_drive")])
    robot = sim.robots["custom"]
    gripper = robot.gripper
    assert gripper is not None and gripper.slot == 1
    assert gripper.get_target() == pytest.approx(0.03)
    grip_qpos = sim.model.joint("custom/grip").qposadr[0]
    initial_position = gripper.get_position()
    sim.data.qpos[grip_qpos] = 0.012
    assert gripper.get_position() == pytest.approx(0.012)
    sim.data.qpos[grip_qpos] = initial_position
    robot.change_controller(create_controller("pd", robot, kp=[1, 1], kd=[0, 0]))
    assert robot.control_joint_names == ("beta", "alpha")
    assert robot.target.position.shape == (2,)
    gripper.set_target(0.04)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.08)
    sim.reset()
    assert gripper.get_target() == pytest.approx(0.03)
    assert not gripper.is_active()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.06)

    with pytest.raises(ValueError, match="position-servo"):
        Simulator(scene, robots=[RobotSpec("bad", "mini", gripper_actuator="beta_drive")])
    path.write_text(
        path.read_text()
        .replace("</actuator>", '<motor name="grip_extra" joint="grip"/></actuator>')
        .replace('ctrl="0 .06 0"', 'ctrl="0 .06 0 0"')
    )
    shared = Simulator(scene, robots=[RobotSpec("shared", "mini", gripper_actuator="grip_drive")])
    with pytest.raises(ValueError, match="also drive the configured gripper joint"):
        create_controller("pd", shared.robots["shared"], kp=[1, 1], kd=[0, 0])
