"""Native actuator tests for controllers on a non-bundled robot."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_SCENES
from mujoco_lab.control import ControlTarget, create_controller
from mujoco_lab.rendering.annotations import (
    NEGATIVE_TORQUE_RGBA,
    POSITIVE_TORQUE_RGBA,
    draw_joint_torques,
)


def small_robot(tmp_path, monkeypatch, mode):
    if mode == "pd":
        actuators = """
            <motor name="beta_drive" joint="beta" gear="-2"/>
            <position name="grip_drive" joint="grip" kp="20"/>
            <motor name="alpha_drive" joint="alpha" gear="3"
                   ctrllimited="true" ctrlrange="-2 2"/>
        """
    else:
        actuators = """
            <position name="beta_drive" joint="beta" kp="40" gear="-2"/>
            <position name="grip_drive" joint="grip" kp="20"/>
            <position name="alpha_drive" joint="alpha" kp="30" gear="3"
                      ctrllimited="true" ctrlrange="-2 2"/>
        """
    path = tmp_path / "mini.xml"
    path.write_text(f"""
        <mujoco model="mini">
          <worldbody>
            <body name="base">
              <body name="first">
                <joint name="alpha" type="hinge" axis="0 0 1"/>
                <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/>
                <body name="middle" pos=".2 0 0">
                  <joint name="passive" type="hinge" axis="0 0 1"/>
                  <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/>
                  <body name="last" pos=".2 0 0">
                    <joint name="beta" type="hinge" axis="0 0 1"/>
                    <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/>
                    <site name="ee_site" pos=".2 0 0"/>
                    <body name="finger" pos=".2 0 0">
                      <joint name="grip" type="slide" axis="0 1 0"/>
                      <geom type="sphere" size=".02" mass=".1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>{actuators}</actuator>
        </mujoco>
    """)
    monkeypatch.setitem(ROBOT_SCENES, "mini", path)
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("custom", "mini")])
    return sim, sim.robots["custom"]


def test_pd_controls_shuffled_geared_motors_and_keeps_passive_joint(tmp_path, monkeypatch):
    sim, robot = small_robot(tmp_path, monkeypatch, "pd")
    controller = create_controller("pd", robot, kp=[20, 10], kd=[0, 0], gravity_compensation=False)
    robot.change_controller(controller)
    assert controller.state.joint_names == ("beta", "alpha")
    assert robot.target.position.shape == (2,)
    robot.target = ControlTarget([0.2, -0.1])
    assert not robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-2, 0, -1 / 3])
    controller.set_gripper_target(0.08)  # Unbounded auxiliary servo.
    assert not robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-2, 0.08, -1 / 3])
    mujoco.mj_forward(sim.model, sim.data)
    dofs = sim.model.jnt_dofadr[controller.state.joint_ids]
    np.testing.assert_allclose(sim.data.qfrc_actuator[dofs], [4, -1])
    passive = sim.model.joint("custom/passive").id
    assert sim.data.qfrc_actuator[sim.model.jnt_dofadr[passive]] == 0
    scene = mujoco.MjvScene(sim.model, maxgeom=10)
    scene.ngeom = 0
    draw_joint_torques(scene, robot, sim.data)
    assert scene.ngeom == 2
    np.testing.assert_allclose(scene.geoms[0].rgba, POSITIVE_TORQUE_RGBA)
    np.testing.assert_allclose(scene.geoms[1].rgba, NEGATIVE_TORQUE_RGBA)
    robot.target = ControlTarget([0.2, 1.0])
    assert robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-2, 0.08, 2])
    sim.reset()
    assert controller.gripper_target == 0
    assert robot.target.position.shape == (2,)


def test_position_commands_drive_shuffled_geared_servos(tmp_path, monkeypatch):
    sim, robot = small_robot(tmp_path, monkeypatch, "position")
    controller = create_controller("position", robot, gravity_compensation=False)
    robot.change_controller(controller)
    assert controller.state.joint_names == ("beta", "grip", "alpha")
    robot.target = ControlTarget([0.2, 0.04, -0.1])
    assert not robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-0.4, 0.04, -0.3])
    mujoco.mj_forward(sim.model, sim.data)
    dofs = sim.model.jnt_dofadr[controller.state.joint_ids]
    np.testing.assert_allclose(sim.data.qfrc_actuator[dofs], [32, 0.8, -27])
    assert controller.state.nv == 3 and robot.state.nv == 4
    sim.run_steps(50)
    assert sim.data.joint("custom/beta").qpos[0] > 0
    with pytest.raises(ValueError, match="torque/force actuators"):
        create_controller("pd", robot, kp=[1, 1, 1], kd=[1, 1, 1])
