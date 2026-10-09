"""Check coupled torque feasibility and the example's linear dynamics solve."""

import runpy
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/pose_acceleration.py"))
acceleration_limits = EXAMPLE["acceleration_limits"]
solve_acceleration = EXAMPLE["solve_acceleration"]


def test_coupled_bounds_cover_all_acceleration_signs():
    mass = np.array([[2.0, 1.0], [1.0, 2.0]])
    bias = np.array([1.0, -0.5])
    torque_limits = np.array([4.0, 3.0])
    result = acceleration_limits(mass, bias, torque_limits)
    assert result.can_hold
    np.testing.assert_allclose(result.isolated, [1.5, 1.25])
    signs = np.array(list(product((-1.0, 1.0), repeat=2)))
    corners = signs * result.simultaneous
    torques = corners @ mass.T + bias
    assert np.all(np.abs(torques) <= torque_limits)
    # Combining isolated limits would exceed a torque limit in this coupled system.
    assert np.any(np.abs((signs * result.isolated) @ mass.T + bias) > torque_limits)
    recovered = solve_acceleration(mass, bias[:, None], torques.T).T
    np.testing.assert_allclose(recovered, corners)


def test_fully_used_holding_torque_leaves_other_decoupled_joint_free():
    result = acceleration_limits(np.diag([2.0, 4.0]), np.array([1.0, 0.0]), np.array([1.0, 4.0]))
    assert result.can_hold
    np.testing.assert_allclose(result.simultaneous, [0.0, 1.0])


def test_unholdable_pose_has_no_symmetric_acceleration_box():
    result = acceleration_limits(np.eye(2), np.array([2.0, 0.0]), np.ones(2))
    assert not result.can_hold
    np.testing.assert_array_equal(result.simultaneous, [0.0, 0.0])


def test_linear_solve_matches_mujoco_forward_dynamics():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <default><geom contype="0" conaffinity="0"/></default>
      <worldbody><body name="first">
        <joint axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="2"/>
        <body name="second" pos=".3 0 0">
          <joint axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/>
        </body>
      </body></worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[:] = [0.2, -0.4]
    data.qvel[:] = [0.1, -0.2]
    torque = np.array([1.0, -0.5])
    data.qfrc_applied[:] = torque
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    acceleration = solve_acceleration(mass, data.qfrc_bias - data.qfrc_passive, torque)
    np.testing.assert_allclose(acceleration, data.qacc, atol=1e-10)
