"""Native actuator tests for controllers on a non-bundled robot."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.control import ControlTarget, create_controller
from mujoco_lab.rendering.annotations import (
    NEGATIVE_TORQUE_RGBA,
    POSITIVE_TORQUE_RGBA,
    draw_joint_torques,
)
from mujoco_lab.state import RobotState


def small_robot(tmp_path, monkeypatch, mode, *, configure_gripper=True):
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
    monkeypatch.setitem(ROBOT_ASSETS, "mini", RobotAsset(path))
    sim = Simulator(
        create_environment("empty"),
        robots=[
            RobotSpec(
                "custom",
                "mini",
                gripper_actuator="grip_drive" if configure_gripper else None,
            )
        ],
    )
    return sim, sim.robots["custom"]


def test_pd_controls_shuffled_geared_motors_and_keeps_passive_joint(tmp_path, monkeypatch):
    sim, robot = small_robot(tmp_path, monkeypatch, "pd")
    controller = create_controller("pd", robot, kp=[20, 10], kd=[0, 0], gravity_compensation=False)
    robot.change_controller(controller)
    assert robot.control_joint_names == ("beta", "alpha")
    assert robot.target.position.shape == (2,)
    robot.target = ControlTarget([0.2, -0.1])
    assert not robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-2, 0, -1 / 3])
    robot.gripper.set_target(0.08)  # Unbounded auxiliary servo.
    assert not robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-2, 0.08, -1 / 3])
    mujoco.mj_forward(sim.model, sim.data)
    dofs = [
        sim.model.jnt_dofadr[sim.model.joint(f"custom/{name}").id]
        for name in robot.control_joint_names
    ]
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
    assert robot.gripper.get_target() == 0
    assert robot.target.position.shape == (2,)


def test_pd_mixed_unconfigured_robot_preserves_native_servo_slot(tmp_path, monkeypatch):
    sim, robot = small_robot(tmp_path, monkeypatch, "pd", configure_gripper=False)
    assert robot.gripper is None
    sim.data.ctrl[robot.actuator_ids] = [0, 0.17, 0]
    robot.change_controller(
        create_controller("pd", robot, kp=[20, 10], kd=[0, 0], gravity_compensation=False)
    )
    assert robot.control_joint_names == ("beta", "alpha")
    robot.target = ControlTarget([0.2, -0.1])
    assert not robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-2, 0.17, -1 / 3])


def test_unsupported_joint_controller_rejects_non_joint_transmission(tmp_path, monkeypatch):
    path = tmp_path / "tendon.xml"
    path.write_text("""
        <mujoco model="tendon_robot">
          <worldbody>
            <body name="base"><joint name="hinge" type="hinge"/>
              <geom type="sphere" size=".1" mass="1"/>
            </body>
          </worldbody>
          <tendon><fixed name="spring"><joint joint="hinge" coef="1"/></fixed></tendon>
          <actuator><motor name="tendon_motor" tendon="spring"/></actuator>
        </mujoco>
    """)
    monkeypatch.setitem(ROBOT_ASSETS, "tendon_robot", RobotAsset(path))
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("tendon", "tendon_robot")])
    robot = sim.robots["tendon"]
    assert robot.control_joint_names == ()
    assert robot.controller is None
    sim.data.ctrl[robot.actuator_ids[0]] = 0.3
    assert not robot.control()
    assert sim.run_steps(1)[robot.name].steps == 1
    assert sim.data.ctrl[robot.actuator_ids[0]] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="direct joint transmissions"):
        create_controller("pd", robot, kp=[1], kd=[1])


def test_uncontrolled_robot_without_actuators_remains_simulatable(tmp_path, monkeypatch):
    path = tmp_path / "passive.xml"
    path.write_text("""
        <mujoco model="passive_robot">
          <worldbody><body name="base"><joint name="hinge" type="hinge"/>
            <geom type="sphere" size=".1" mass="1"/>
          </body></worldbody>
        </mujoco>
    """)
    monkeypatch.setitem(ROBOT_ASSETS, "passive", RobotAsset(path))
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("passive", "passive")])
    robot = sim.robots["passive"]
    assert robot.control_joint_names == ()
    assert not robot.control()
    assert sim.run_steps(1)[robot.name].steps == 1
    with pytest.raises(ValueError, match="joint actuators"):
        create_controller("pd", robot, kp=[1], kd=[1])


def test_position_commands_drive_shuffled_geared_servos(tmp_path, monkeypatch):
    sim, robot = small_robot(tmp_path, monkeypatch, "position")
    controller = create_controller("position", robot, gravity_compensation=False)
    robot.change_controller(controller)
    assert robot.control_joint_names == ("beta", "alpha")
    robot.target = ControlTarget([0.2, -0.1])
    robot.gripper.set_target(0.04)
    assert not robot.control()
    np.testing.assert_allclose(sim.data.ctrl[robot.actuator_ids], [-0.4, 0.04, -0.3])
    mujoco.mj_forward(sim.model, sim.data)
    dofs = [
        sim.model.jnt_dofadr[sim.model.joint(f"custom/{name}").id]
        for name in robot.control_joint_names
    ]
    np.testing.assert_allclose(sim.data.qfrc_actuator[dofs], [32, -27])
    assert len(robot.control_joint_names) == 2 and robot.state.nv == 4
    sim.run_steps(50)
    assert sim.data.joint("custom/beta").qpos[0] > 0
    with pytest.raises(ValueError, match="torque/force actuators"):
        create_controller("pd", robot, kp=[1, 1, 1], kd=[1, 1, 1])


def test_robot_selects_arm_order_without_cloning_robot_state(tmp_path, monkeypatch):
    sim, robot = small_robot(tmp_path, monkeypatch, "pd")

    def unexpected_state_init(*args, **kwargs):
        raise AssertionError("Controller construction must reuse Robot.state")

    with monkeypatch.context() as context:
        context.setattr(RobotState, "__init__", unexpected_state_init)
        controller = create_controller(
            "pd", robot, kp=[20, 10], kd=[0, 0], gravity_compensation=False
        )
    assert robot.controller is None and robot.target is None
    assert robot.control_joint_names == ()
    assert controller._owner is robot.state
    assert robot.control_qpos_indices == []
    robot.change_controller(controller)
    assert robot.control_joint_names == ("beta", "alpha")
    qpos_indices = robot.control_qpos_indices
    joint_slots = robot.control_joint_slots
    assert qpos_indices == [
        sim.model.joint("custom/beta").qposadr[0],
        sim.model.joint("custom/alpha").qposadr[0],
    ]
    assert joint_slots == [2, 0]
    np.testing.assert_allclose(
        robot.state.get_jacobian("ee_site", joint_slots),
        robot.state.get_jacobian("ee_site")[:, joint_slots],
    )
    np.testing.assert_allclose(
        robot.state.get_mass_matrix(joint_slots),
        robot.state.get_mass_matrix()[np.ix_(joint_slots, joint_slots)],
    )
    assert controller._owner is robot.state
    assert robot.get_control_state().qpos.tolist() == robot.state.snapshot().qpos[[2, 0]].tolist()


def test_robot_control_uses_cached_selected_joint_snapshot(tmp_path, monkeypatch):
    sim, robot = small_robot(tmp_path, monkeypatch, "pd")
    controller = create_controller("pd", robot, kp=[20, 10], kd=[0, 0], gravity_compensation=False)
    robot.change_controller(controller)
    target = ControlTarget([0.2, -0.1])
    robot.target = target
    snapshot = robot.get_control_state()
    expected = controller.compute(snapshot, target)
    robot.control()
    command_before = sim.data.ctrl[robot.actuator_ids].copy()
    sim.data.qpos[robot.state.qpos_indices] += 0.4
    sim.data.qvel[robot.state.dof_indices] += 0.5
    np.testing.assert_array_equal(controller.compute(robot.get_control_state(), target), expected)
    robot.control()
    np.testing.assert_array_equal(sim.data.ctrl[robot.actuator_ids], command_before)
    robot.update_state()
    assert not np.array_equal(robot.get_control_state().qpos, snapshot.qpos)
    robot.control()
    assert not np.array_equal(sim.data.ctrl[robot.actuator_ids], command_before)
