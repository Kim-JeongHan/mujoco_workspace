"""Load MuJoCo models and initialize their native simulation state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mujoco_lab.assets import ASSET_ROOT

if TYPE_CHECKING:
    import mujoco

__all__ = ["ASSET_ROOT", "initialize_data", "load_simulation"]


def initialize_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Create state, apply the home keyframe when present, and update derived fields."""
    import mujoco

    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    return data


def load_simulation(model_path: str | Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load an MJCF/URDF file and initialize its derived state.

    Relative includes and mesh paths are resolved by MuJoCo against the model file.
    If present, the named ``home`` keyframe initializes both state and controls.
    """
    import mujoco

    path = Path(model_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, initialize_data(model)
