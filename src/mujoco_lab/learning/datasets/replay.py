"""Capture full-scene geometry for playback, independently of observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import mujoco
import numpy as np

from mujoco_lab.simulation import Simulator


def model_signature(model: mujoco.MjModel) -> str:
    """Fingerprint the compiled model, including geometry and joint ordering."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer).hexdigest()


def visual_signature(model: mujoco.MjModel) -> str:
    """Fingerprint the qpos layout and visible scene data for v2 playback."""
    if model.nflex:
        raise ValueError("Replay visual signature does not support flex geometry")
    fields = [
        "qpos0",
        "body_parentid",
        "body_pos",
        "body_quat",
        "body_mocapid",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_pos",
        "jnt_axis",
        "geom_bodyid",
        "geom_type",
        "geom_dataid",
        "geom_matid",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_rgba",
        "geom_group",
        "site_bodyid",
        "site_type",
        "site_dataid",
        "site_matid",
        "site_pos",
        "site_quat",
        "site_size",
        "site_rgba",
        "site_group",
        "mesh_pos",
        "mesh_quat",
        "mesh_scale",
        "mesh_vertadr",
        "mesh_vertnum",
        "mesh_faceadr",
        "mesh_facenum",
        "mesh_vert",
        "mesh_face",
        "mesh_normaladr",
        "mesh_normalnum",
        "mesh_normal",
        "mesh_facenormal",
        "mesh_texcoordadr",
        "mesh_texcoordnum",
        "mesh_texcoord",
        "mesh_facetexcoord",
        "hfield_adr",
        "hfield_nrow",
        "hfield_ncol",
        "hfield_size",
        "hfield_data",
        "mat_rgba",
        "mat_emission",
        "mat_specular",
        "mat_shininess",
        "mat_reflectance",
        "mat_metallic",
        "mat_roughness",
        "mat_texid",
        "mat_texrepeat",
        "mat_texuniform",
        "tex_type",
        "tex_colorspace",
        "tex_width",
        "tex_height",
        "tex_nchannel",
        "tex_adr",
        "tex_data",
        "light_bodyid",
        "light_targetbodyid",
        "light_mode",
        "light_type",
        "light_pos",
        "light_dir",
        "light_ambient",
        "light_diffuse",
        "light_specular",
        "light_active",
        "light_attenuation",
        "light_bulbradius",
        "light_castshadow",
        "light_cutoff",
        "light_exponent",
        "light_intensity",
        "light_range",
        "light_softness",
        "light_texid",
        "skin_vertadr",
        "skin_vertnum",
        "skin_vert",
        "skin_faceadr",
        "skin_facenum",
        "skin_face",
        "skin_boneadr",
        "skin_bonenum",
        "skin_bonebodyid",
        "skin_bonebindpos",
        "skin_bonebindquat",
        "skin_bonevertadr",
        "skin_bonevertnum",
        "skin_bonevertid",
        "skin_bonevertweight",
        "skin_texcoordadr",
        "skin_texcoord",
        "skin_matid",
        "skin_rgba",
        "skin_group",
        "skin_inflate",
    ]
    digest = hashlib.sha256()
    for kind, count in (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
    ):
        names = [mujoco.mj_id2name(model, kind, index) for index in range(count)]
        digest.update(json.dumps(names).encode())
    for name in fields:
        value = np.asarray(getattr(model, name))
        digest.update(f"{name}:{value.shape}:{value.dtype}:".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def capture_frame(simulator: Simulator) -> dict[str, np.ndarray]:
    """Copy one frame; learning observations cannot substitute for global qpos."""
    data = simulator.data
    return {
        "qpos": data.qpos.copy(),
        "frame_times": np.asarray(data.time),
        "mocap_pos": data.mocap_pos.copy(),
        "mocap_quat": data.mocap_quat.copy(),
    }


def replay_action_repeat(metadata: Mapping[str, Any]) -> int:
    """Return recorded physics ticks per action; legacy recordings use one."""
    repeat = metadata.get("physics_steps_per_action", 1)
    if isinstance(repeat, bool) or not isinstance(repeat, Integral) or repeat <= 0:
        raise ValueError("physics_steps_per_action must be a positive integer")
    return int(repeat)


def replay_cube_yaw_range_degrees(metadata: Mapping[str, Any]) -> float:
    """Return the recorded yaw range; legacy recordings used zero yaw."""
    value = metadata.get("cube_yaw_range_degrees", 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("cube_yaw_range_degrees must be finite and nonnegative")
    return float(value)


def cube_stack_metadata(
    simulator: Simulator,
    *,
    cubes: int,
    robot: str,
    physics_steps_per_action: int = 1,
    cube_yaw_range_degrees: float = 0.0,
) -> dict[str, Any]:
    """Describe a bundled cube scene with one robot at its scene mount.

    New v2 recordings use a visual compatibility signature. Legacy v1 replay
    still requires the original MuJoCo version and full model binary hash.
    """
    repeat = replay_action_repeat({"physics_steps_per_action": physics_steps_per_action})
    yaw_range = replay_cube_yaw_range_degrees(
        {"cube_yaw_range_degrees": cube_yaw_range_degrees}
    )
    return {
        "schema_version": 2,
        "scene": "cube_stack",
        "environment": "table_shelf",
        "cubes": cubes,
        "robot": robot,
        "robot_name": next(iter(simulator.robots)),
        "dt": simulator.dt,
        "physics_steps_per_action": repeat,
        "cube_yaw_range_degrees": yaw_range,
        "mujoco_version": mujoco.__version__,
        "model_sha256": model_signature(simulator.model),
        "visual_sha256": visual_signature(simulator.model),
    }
