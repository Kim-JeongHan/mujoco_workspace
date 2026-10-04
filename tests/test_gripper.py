"""Robot-owned gripper control alongside selected joint controllers."""

from dataclasses import replace

import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import GripperConfig


def test_forte_gripper_controls_without_arm_controller_and_resets_home():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
    )
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
        gripper.set_target(-0.038)
    with pytest.raises(ValueError, match="gripper target"):
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
        assert sim.data.ctrl[gripper.actuator_id] == boundary * gripper.gear
        with pytest.raises(ValueError, match="gripper target"):
            gripper.set_target(boundary + direction * 1e-10)
        assert gripper.get_target() == boundary


def test_panda_arm_position_target_excludes_gripper_and_preserves_gripper_override():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "panda", config=load_robot_config("panda"))],
    )
    robot = sim.robots["robot"]
    gripper = robot.gripper
    assert gripper is not None
    robot.change_controller(create_test_controller(robot, controller="position"))
    assert robot.target.position.shape == (7,)
    assert len(robot.control_joint_names) == 7
    gripper.set_target(0.02)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.02)
    robot.change_controller(None)
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.02)
    robot.change_controller(create_test_controller(robot, controller="position"))
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(0.02)
    sim.reset()
    assert not gripper.is_active()
    home_ctrl = sim.data.ctrl[gripper.actuator_id]
    sim.step()
    assert sim.data.ctrl[gripper.actuator_id] == pytest.approx(home_ctrl)


def test_forte_full_closure_reduces_pad_gap_by_74_mm_without_self_contact():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
    )
    robot = sim.robots["robot"]
    robot.change_controller(create_test_controller(robot, controller="pd", frame="grasp"))
    pads = [sim.model.geom(f"robot/gripper_{side}_pad").id for side in ("left", "right")]

    def pad_gap():
        first, second = pads
        axis = sim.data.geom_xmat[first].reshape(3, 3)[:, 0]
        separation = abs((sim.data.geom_xpos[second] - sim.data.geom_xpos[first]) @ axis)
        return separation - sum(sim.model.geom_size[g, 0] for g in pads)

    open_gap = pad_gap()
    robot.gripper.set_target(-0.037)
    sim.run_steps(2000)
    closed_gap = pad_gap()
    np.testing.assert_allclose(sim.data.qpos[robot.state.qpos_indices[7:]], -0.037, atol=1e-6)
    assert open_gap == pytest.approx(0.075396278, abs=1e-8)
    assert closed_gap == pytest.approx(0.001396278, abs=1e-8)
    assert open_gap - closed_gap == pytest.approx(0.074, abs=1e-8)
    assert sim.data.ncon == 0
    assert not sim.data.warning.number.any()


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
    (tmp_path / "robot.yaml").write_text("""controller:
  name: position
gripper:
  actuator: left_drive
""")
    monkeypatch.setitem(ROBOT_ASSETS, "two_grippers", RobotAsset(path))
    override_config = replace(
        load_robot_config("two_grippers"), gripper=GripperConfig(actuator="right_drive")
    )
    scene = create_environment("empty")
    default = Simulator(
        scene, robots=[RobotSpec("robot", "two_grippers", config=load_robot_config("two_grippers"))]
    ).robots["robot"]
    override = Simulator(
        scene, robots=[RobotSpec("robot", "two_grippers", config=override_config)]
    ).robots["robot"]
    assert default.gripper is not None and default.gripper.slot == 0
    assert override.gripper is not None and override.gripper.slot == 1
    disabled = Simulator(
        scene,
        robots=[RobotSpec("robot", "two_grippers", config=replace(override_config, gripper=None))],
    ).robots["robot"]
    assert disabled.gripper is None


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
    (tmp_path / "robot.yaml").write_text("""controller:
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

    config = replace(load_robot_config("mini"), gripper=GripperConfig(actuator="grip_drive"))
    sim = Simulator(scene, robots=[RobotSpec("custom", "mini", config=config)])
    robot = sim.robots["custom"]
    gripper = robot.gripper
    assert gripper is not None and gripper.slot == 1
    assert gripper.get_target() == pytest.approx(0.03)
    grip_qpos = sim.model.joint("custom/grip").qposadr[0]
    initial_position = gripper.get_position()
    sim.data.qpos[grip_qpos] = 0.012
    assert gripper.get_position() == pytest.approx(0.012)
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

    with pytest.raises(ValueError, match="position-servo"):
        bad_config = replace(config, gripper=GripperConfig(actuator="beta_drive"))
        Simulator(scene, robots=[RobotSpec("bad", "mini", config=bad_config)])
    path.write_text(
        path.read_text()
        .replace("</actuator>", '<motor name="grip_extra" joint="grip"/></actuator>')
        .replace('ctrl="0 .06 0"', 'ctrl="0 .06 0 0"')
    )
    shared = Simulator(scene, robots=[RobotSpec("shared", "mini", config=config)])
    with pytest.raises(ValueError, match="also drive the configured gripper joint"):
        create_test_controller(shared.robots["shared"], controller="pd")
