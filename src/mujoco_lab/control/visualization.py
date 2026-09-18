"""MuJoCo annotations extracted from the Forte demos and workspace frame adapter."""

import mujoco
import numpy as np

from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD

POSITIVE_TORQUE_RGBA = (0.20, 0.85, 0.35, 0.95)
NEGATIVE_TORQUE_RGBA = (0.95, 0.45, 0.10, 0.95)
TARGET_RGBA = (0.20, 0.90, 0.30, 0.35)
ACTUAL_RGBA = (1.00, 0.25, 0.15, 0.95)
FORCE_RGBA = (0.25, 0.55, 1.00, 0.95)
ERROR_RGBA = (1.00, 0.85, 0.10, 0.90)
PATH_RGBA = (0.55, 0.65, 0.80, 0.35)

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


def draw_joint_torques(scene, model, data, torques) -> None:
    limits = np.maximum(np.abs(model.actuator_ctrlrange).max(axis=1), 1e-6)
    for index, torque in enumerate(torques):
        joint_id = model.actuator_trnid[index, 0]
        reference = TORQUE_ARROW_REFERENCE * limits[index]
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


def annotate_controller(scene, model, data, controller) -> None:
    """Draw reference markers without coupling controller math to a renderer."""
    if isinstance(controller, JointSpacePD):
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = controller._desired
        mujoco.mj_kinematics(model, scratch)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        add_marker(scene, scratch.site_xpos[site], TARGET_RGBA, 0.028)
        add_marker(scene, data.site_xpos[site], ACTUAL_RGBA, 0.016)
    elif isinstance(controller, OperationalSpaceControl):
        for index in range(36):
            position, _, _ = controller.circle(controller.period * index / 36)
            add_marker(scene, position, PATH_RGBA, 0.005)
        actual = controller.dynamics.get_frame_position(controller.frame)
        add_marker(scene, controller._target, TARGET_RGBA, 0.032)
        add_marker(scene, actual, ACTUAL_RGBA, 0.014)
        add_arrow(
            scene,
            actual,
            scaled_vector(
                controller._target - actual,
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


def annotate(scene, model, data, controller) -> None:
    """Draw torque arrows and controller annotations into a MuJoCo scene."""
    draw_joint_torques(scene, model, data, data.ctrl)
    annotate_controller(scene, model, data, controller)
