"""Computed-torque dynamics, joint ordering, and acceleration-gain configuration."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.robot.robot import Constraints, ControllerConfig, PoseConfig, RobotConfig
from mujoco_lab.control import JointSpaceComputedTorque, create_controller
from mujoco_lab.trajectory_generation import ControlTarget


@pytest.fixture
def coupled_arm(tmp_path, monkeypatch):
    path = tmp_path / "coupled.xml"
    path.write_text("""<mujoco>
      <default><geom contype="0" conaffinity="0"/></default>
      <worldbody><body name="base"><body name="first">
        <joint name="alpha" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="2"/>
        <body name="second" pos=".3 0 0">
          <joint name="beta" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/>
          <site name="grasp" pos=".2 0 0"/>
        </body>
      </body></body></worldbody>
      <actuator>
        <motor name="beta_drive" joint="beta" gear="-2"/>
        <motor name="alpha_drive" joint="alpha" gear="3"/>
      </actuator>
    </mujoco>""")
    monkeypatch.setitem(ROBOT_ASSETS, "coupled", RobotAsset(path))
    config = RobotConfig(
        ControllerConfig("computed_torque"),
        Constraints(["alpha", "beta"], [[-np.pi, np.pi]] * 2, [1.0, 2.0], [3.0, 4.0]),
        pose=PoseConfig([0.25, -0.35]),
    )
    sim = Simulator(
        create_environment("empty"), robots=[RobotSpec("arm", "coupled", config=config)]
    )
    return sim, sim.robots["arm"]


@pytest.mark.parametrize("with_derivatives", [False, True])
def test_feedback_linearizes_coupled_geared_joint_dynamics(coupled_arm, with_derivatives):
    sim, robot = coupled_arm
    controller = create_controller(
        robot, ControllerConfig("computed_torque", natural_frequency=10, damping_ratio=0.8)
    )
    robot.change_controller(controller)
    assert robot.control_joint_names == ("beta", "alpha")
    sim.data.qvel[robot.state.dof_indices] = [0.1, -0.2]
    mujoco.mj_forward(sim.model, sim.data)
    robot.update_state()
    state = robot.get_control_state()
    error = np.array([0.1, -0.05])
    velocity = np.array([0.3, -0.1]) if with_derivatives else None
    acceleration = np.array([0.7, -0.4]) if with_derivatives else None
    robot.target = ControlTarget(state.qpos + error, velocity, acceleration)
    command = 100 * error + 16 * ((0 if velocity is None else velocity) - state.qvel)
    if acceleration is not None:
        command += acceleration
    mass = robot.state.get_mass_matrix(robot.control_joint_slots)
    assert abs(mass[0, 1]) > 0.01  # Nonzero coupling must contribute to torque.
    assert not robot.control()
    mujoco.mj_forward(sim.model, sim.data)
    dofs = [robot.state.dof_indices[slot] for slot in robot.control_joint_slots]
    np.testing.assert_allclose(sim.data.qacc[dofs], command, atol=1e-12)
    np.testing.assert_allclose(sim.data.qfrc_actuator[dofs], mass @ command + state.bias_forces)
    assert controller.get_tracking_error() == pytest.approx(0.1)
    controller.reset()
    assert controller.get_tracking_error() == 0


def test_per_joint_acceleration_gains_and_optional_bias(coupled_arm):
    _, robot = coupled_arm
    controller = JointSpaceComputedTorque(
        robot.state, [1, 0], [10, 20], [1, 0.5], gravity_compensation=False
    )
    robot.change_controller(controller)
    state = robot.get_control_state()
    target = ControlTarget(state.qpos + [0.1, -0.05], acceleration=np.array([0.5, 0.2]))
    np.testing.assert_allclose(
        controller.compute(state, target),
        robot.state.get_mass_matrix([1, 0]) @ np.array([10.5, -19.8]),
    )


def test_target_joint_count_cannot_silently_broadcast(coupled_arm):
    _, robot = coupled_arm
    controller = create_controller(robot, ControllerConfig("computed_torque"))
    with pytest.raises(ValueError, match="one value per controlled joint"):
        controller.compute(robot.joint_state, ControlTarget(np.zeros(1)))
