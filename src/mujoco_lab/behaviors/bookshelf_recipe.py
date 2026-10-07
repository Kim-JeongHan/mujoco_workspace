"""Validated robot recipes for the fixed bookshelf insertion sequence."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mujoco_lab.assets import RobotName
from mujoco_lab.control.trajectory import MotionRatio

STAGE_ORDER = (
    "approach",
    "pick",
    "close",
    "lift",
    "preinsert",
    "insert",
    "lower",
    "release",
    "retract",
)


class StageRecipe(BaseModel):
    """One named book stage with motion ratios and physical grasp checks."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str
    gripper_mode: Literal[0, 1]  # 1 closes; 0 opens.
    arm: MotionRatio | None = None
    gripper: MotionRatio | None = None


class BookshelfRecipe(BaseModel):
    """Fixed nine-stage workflow and robot-specific insertion parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    arm: MotionRatio = (1.0, 1.0)
    gripper: MotionRatio = (1.0, 1.0)
    waypoint_tolerance: float = Field(gt=0)
    arm_tolerance: float = Field(gt=0)
    stage_dwell_s: float = Field(ge=0)
    stage_timeout_s: float = Field(gt=0)
    lost_grasp_grace_s: float = Field(ge=0)
    approach_distance_m: float = Field(gt=0)
    lift_height_m: float = Field(gt=0)
    min_lift_height_m: float = Field(gt=0)
    insertion_clearance_m: float = Field(gt=0)
    shelf_front_y_m: float
    preinsert_clearance_m: float = Field(gt=0)
    retract_distance_m: float = Field(gt=0)
    stages: tuple[StageRecipe, ...]

    @model_validator(mode="after")
    def validate_workflow(self) -> BookshelfRecipe:
        if tuple(stage.name for stage in self.stages) != STAGE_ORDER:
            raise ValueError("stages must use the ordered nine-stage bookshelf sequence")
        if self.min_lift_height_m >= self.lift_height_m:
            raise ValueError("min_lift_height_m must be less than lift_height_m")
        return self


def load_recipe(robot_type: RobotName) -> BookshelfRecipe:
    """Read and validate the bundled bookshelf recipe for one robot."""
    path = files("mujoco_lab.behaviors").joinpath("recipes", "bookshelf", f"{robot_type}.yaml")
    return BookshelfRecipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
