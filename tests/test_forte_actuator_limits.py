"""Verify rated Forte limits reach MuJoCo's native joint actuation."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.utils import Transform


@pytest.mark.parametrize("asset", ["forte", "forte_backup"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_native_motor_torque_is_clipped_to_rated_joint_limits(asset, direction):
    path = Path(__file__).resolve().parents[1] / "src/mujoco_lab/assets/robot" / asset / "robot.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    joints = (
        "shoulder_yaw",
        "shoulder_pitch",
        "shoulder_roll",
        "elbow_pitch",
        "lower_arm_roll",
        "wrist_pitch",
        "wrist_roll",
    )
    limits = np.array([1.8, 66.0, 11.0, 11.0, 3.6, 3.6, 1.8])
    actuators = [model.actuator(f"{name}_motor").id for name in joints]
    dofs = [int(model.joint(name).dofadr[0]) for name in joints]
    data.ctrl[actuators] = direction * 1000
    data.qvel[dofs] = 0.3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[dofs], direction * limits)
    gears = np.array([1, 6, 1, 1, 2, 2, 1])
    np.testing.assert_array_equal(model.actuator_gear[actuators, 0], gears)
    np.testing.assert_allclose(data.actuator_force[actuators], direction * limits / gears)
    np.testing.assert_allclose(data.actuator_velocity[actuators], 0.3 * gears)
    # This servo applies linear force, so its N range is not replaced with motor N.m.
    gripper = model.actuator("gripper_motor").id
    np.testing.assert_array_equal(model.actuator_forcerange[gripper], [-20, 20])


@pytest.mark.parametrize("pose", ["zero", "default"])
@pytest.mark.parametrize("width", [0.0, 0.074])
def test_operating_acceleration_limits_fit_rated_torques_at_reference_poses(pose, width):
    """A zero-pose acceleration maximum must not overload the folded reference pose."""
    config = load_robot_config("forte")
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("arm", "forte", Transform(translation=[0, 0, 0.8]), config=config)],
    )
    robot = sim.robots["arm"]
    constraints = config.constraints
    gripper = robot.gripper
    assert constraints is not None
    assert gripper is not None
    names = constraints.joint_names
    slots = [robot.state.joint_names.index(name) for name in names]
    positions = [robot.state.qpos_indices[slot] for slot in slots]
    dofs = [robot.state.dof_indices[slot] for slot in slots]
    sim.data.qpos[positions] = config.pose.named_poses()[pose][: len(names)]
    gripper.apply_initial_width(width)
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)
    mass = robot.state.get_mass_matrix(slots)
    bias = sim.data.qfrc_bias[dofs] - sim.data.qfrc_passive[dofs]
    actuators = [sim.model.actuator(f"arm/{name}_motor").id for name in names]
    limits = np.min(np.abs(sim.model.actuator_ctrlrange[actuators]), axis=1)
    limits *= np.abs(sim.model.actuator_gear[actuators, 0])
    required = np.abs(bias) + np.abs(mass) @ constraints.acceleration_limit
    assert np.all(required <= limits)
