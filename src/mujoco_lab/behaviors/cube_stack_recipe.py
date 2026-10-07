"""Validated robot recipes for the fixed cube stacking sequence."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mujoco_lab.assets import RobotName
from mujoco_lab.control.trajectory import MotionRatio

STAGE_ORDER = ("above_pick", "pick", "close", "lift", "above_place", "place", "release", "retract")


class StageRecipe(BaseModel):
    """One named pose and its execution ratios, applied once per cube."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str
    offset_xyz_m: tuple[float, float, float]
    gripper_mode: Literal[0, 1]  # 1 closes; 0 opens.
    arm: MotionRatio | None = None
    gripper: MotionRatio | None = None
    min_lift_height_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_physical_checks(self) -> StageRecipe:
        if self.min_lift_height_m is not None and self.name != "lift":
            raise ValueError("min_lift_height_m is only valid for the lift stage")
        return self


class CubeStackRecipe(BaseModel):
    """Fixed eight-stage workflow and robot-specific numeric parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    euler_xyz_degrees: tuple[float, float, float] | None = None
    frame: Literal["grasp"]
    arm: MotionRatio = (1.0, 1.0)
    gripper: MotionRatio = (1.0, 1.0)
    arm_tolerance: float = Field(gt=0)
    gripper_tolerance: float = Field(gt=0)
    lost_grasp_grace_s: float = Field(ge=0)
    stages: tuple[StageRecipe, ...]

    @model_validator(mode="after")
    def validate_workflow(self) -> CubeStackRecipe:
        if tuple(stage.name for stage in self.stages) != STAGE_ORDER:
            raise ValueError("stages must use the ordered eight-stage cube stack sequence")
        return self


def load_recipe(robot_type: RobotName) -> CubeStackRecipe:
    """Read and validate the bundled recipe for one supported robot."""
    path = files("mujoco_lab.behaviors").joinpath("recipes", "cube_stack", f"{robot_type}.yaml")
    return CubeStackRecipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
