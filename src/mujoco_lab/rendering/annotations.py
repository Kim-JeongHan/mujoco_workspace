"""MuJoCo annotations extracted from the Forte demos and workspace frame adapter."""

import mujoco
import numpy as np

from mujoco_lab.control.factory import ActuatorController
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD

POSITIVE_TORQUE_RGBA = (0.20, 0.85, 0.35, 0.95)
NEGATIVE_TORQUE_RGBA = (0.95, 0.45, 0.10, 0.95)
TARGET_RGBA = (0.20, 0.90, 0.30, 0.35)
ACTUAL_RGBA = (1.00, 0.25, 0.15, 0.95)
FORCE_RGBA = (0.25, 0.55, 1.00, 0.95)
ERROR_RGBA = (1.00, 0.85, 0.10, 0.90)

MAX_TORQUE_ARROW = 0.45
ARROW_WIDTH = 0.009
# Full arrow length at 20% of the actuator limit. Typical torques are much
# smaller than 87 Nm, so scaling against the raw limit looks tiny.
TORQUE_ARROW_REFERENCE = 0.2


FORCE_ARROW_REFERENCE = 1.0
MAX_FORCE_ARROW = 0.40
ERROR_ARROW_REFERENCE = 0.01
MAX_ERROR_ARROW = 0.30


def _next_geom(scene):
    if scene.ngeom >= scene.maxgeom:
        return None
    scene.ngeom += 1
    return scene.geoms[scene.ngeom - 1]


def add_arrow(scene, origin, vector, rgba, width: float = ARROW_WIDTH) -> None:
    geom = _next_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    start = np.asarray(origin, dtype=float)
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        start,
        start + np.asarray(vector, dtype=float),
    )


def add_marker(scene, position, rgba, size: float = 0.02) -> None:
    geom = _next_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, size),
        np.asarray(position, dtype=float),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )


def scaled_length(value: float, reference: float, max_length: float) -> float:
    """Map a signed value to an arrow length. Sqrt scaling keeps small
    values visible next to large ones.
    """
    ratio = float(np.clip(value / reference, -1.0, 1.0))
    return max_length * np.sign(ratio) * np.sqrt(abs(ratio))


def scaled_vector(vector, reference: float, max_length: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-9:
        return np.zeros(3)
    return vector * (scaled_length(magnitude, reference, max_length) / magnitude)


def draw_joint_torques(scene, robot, data) -> None:
    model = robot.model
    for actuator_id in robot.actuator_ids:
        # Position servos use ctrl as a target angle/displacement, not torque.
        if model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
            continue
        gain = model.actuator_gainprm[actuator_id, 0]
        gear = model.actuator_gear[actuator_id, 0]
        torque = data.ctrl[actuator_id] * gain * gear
        joint_id = model.actuator_trnid[actuator_id, 0]
        limit = (
            np.abs(model.actuator_ctrlrange[actuator_id]).max()
            if model.actuator_ctrllimited[actuator_id]
            else 1.0
        )
        reference = TORQUE_ARROW_REFERENCE * max(limit * abs(gain * gear), 1e-6)
        length = scaled_length(torque, reference, MAX_TORQUE_ARROW)
        if abs(length) < 1e-3:
            continue
        rgba = POSITIVE_TORQUE_RGBA if torque > 0 else NEGATIVE_TORQUE_RGBA
        add_arrow(
            scene,
            data.xanchor[joint_id],
            data.xaxis[joint_id] * length,
            rgba,
        )


def annotate_controller(scene, robot, data, target_data) -> None:
    """Draw cached targets using rendering data and shared PD kinematics scratch."""
    model, controller = robot.model, robot.controller
    binding = controller if isinstance(controller, ActuatorController) else None
    if binding is not None:
        controller = binding.algorithm
    if isinstance(controller, JointSpacePD):
        if robot.target is None:
            return
        target_data.qpos[:] = data.qpos
        indices = binding.state.qpos_indices if binding is not None else robot.state.qpos_indices
        target_data.qpos[indices] = robot.target.position
        mujoco.mj_kinematics(model, target_data)
        site = robot.state.site_id(binding.frame if binding is not None else "ee_site")
        add_marker(scene, target_data.site_xpos[site], TARGET_RGBA, 0.028)
        add_marker(scene, data.site_xpos[site], ACTUAL_RGBA, 0.016)
    elif isinstance(controller, OperationalSpaceControl):
        actual = data.site_xpos[robot.state.site_id(controller.frame)]
        target = actual if robot.target is None else robot.target.position
        add_marker(scene, target, TARGET_RGBA, 0.032)
        add_marker(scene, actual, ACTUAL_RGBA, 0.014)
        add_arrow(
            scene,
            actual,
            scaled_vector(
                target - actual,
                ERROR_ARROW_REFERENCE,
                MAX_ERROR_ARROW,
            ),
            ERROR_RGBA,
            0.006,
        )
        add_arrow(
            scene,
            actual,
            scaled_vector(controller._force, FORCE_ARROW_REFERENCE, MAX_FORCE_ARROW),
            FORCE_RGBA,
            0.011,
        )


def annotate(scene, simulator, data) -> None:
    """Draw scoped annotations without changing live state or controller caches."""
    target_data = None
    for robot in simulator.robots.values():
        if robot.controller is not None:
            controller = robot.controller
            if isinstance(controller, ActuatorController):
                controller = controller.algorithm
            if isinstance(controller, JointSpacePD) and target_data is None:
                target_data = mujoco.MjData(simulator.model)
                mujoco.mj_copyData(target_data, simulator.model, data)
            draw_joint_torques(scene, robot, data)
            annotate_controller(scene, robot, data, target_data)
