"""Scene settings serialized with recorded demonstration geometry."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class ReplayMetadataConfig:
    """Recorded robot, action cadence, and shared randomization settings."""

    robot: str
    physics_steps_per_action: int = 1
    xy_range: float = 0.02
    gripper_action_units: Literal["opening_width_m"] = field(default="opening_width_m", init=False)


@dataclass(frozen=True, kw_only=True)
class CubeStackReplayConfig(ReplayMetadataConfig):
    """Recorded cube scene and its collection randomization."""

    cubes: int
    cube_yaw_range_degrees: float = 0.0
    min_gap: float = 0.01
    scene: Literal["cube_stack"] = field(default="cube_stack", init=False)
    environment: Literal["table_shelf"] = field(default="table_shelf", init=False)


@dataclass(frozen=True, kw_only=True)
class BookReplayConfig(ReplayMetadataConfig):
    """Recorded book scene and its collection randomization."""

    book: str
    book_yaw_range_degrees: float = 0.0
    scene: Literal["book_insertion"] = field(default="book_insertion", init=False)
    environment: Literal["book_shelf"] = field(default="book_shelf", init=False)
