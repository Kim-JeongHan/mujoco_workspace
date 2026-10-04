"""Typed robot controller settings loaded from robot.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

import mujoco
import yaml
from dacite import from_dict
from pydantic import ConfigDict, Field

# dacite requires a runtime Literal alias rather than a PEP 695 alias.
ControllerType = Literal["none", "pd", "position", "osc"]  # noqa: UP040

LimitValue = Annotated[float, Field(gt=0)]


@dataclass(kw_only=True)
class MotionLimits:
    """Optional velocity and acceleration limits in the controlled joints' units.

    Scalars apply to every joint; lists follow the resolved joint order.
    An omitted value inherits the corresponding robot or recipe limit.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", allow_inf_nan=False)

    velocity_limit: LimitValue | list[LimitValue] | None = None
    acceleration_limit: LimitValue | list[LimitValue] | None = None

    def override(self, other: MotionLimits) -> MotionLimits:
        """Use explicitly supplied limits from another layer, inheriting omissions."""
        return MotionLimits(
            velocity_limit=(
                self.velocity_limit if other.velocity_limit is None else other.velocity_limit
            ),
            acceleration_limit=(
                self.acceleration_limit
                if other.acceleration_limit is None
                else other.acceleration_limit
            ),
        )


@dataclass(frozen=True)
class RobotModelInfo:
    joint_names: tuple[str, ...] = ()
    actuator_names: tuple[str, ...] = ()
    site_names: tuple[str, ...] = ()
    root_name: str = ""


@dataclass
class PDGain:
    kp: float
    kd: float


@dataclass
class FortePDGain:
    shoulder_yaw: PDGain
    shoulder_pitch: PDGain
    shoulder_roll: PDGain
    elbow_pitch: PDGain
    lower_arm_roll: PDGain
    wrist_pitch: PDGain
    wrist_roll: PDGain


@dataclass
class Constraints:
    """Joint bounds and motion limits in named-joint order, using native units."""

    joint_names: list[str]
    position_limit: list[list[float]]  # [lower, upper], in radians or meters.
    velocity_limit: list[float]
    acceleration_limit: list[float]

    def get_info(self, index: int) -> tuple[str, list[float], list[float], list[float]]:
        """Return joint name, position limits, velocity limits, and acceleration limits."""
        return (
            self.joint_names[index],
            self.position_limit[index],
            [self.velocity_limit[index]],
            [self.acceleration_limit[index]],
        )


@dataclass
class GripperConfig:
    actuator: str
    velocity_limit: float = field(default=0.2, kw_only=True)  # Gripper joint speed in m/s.
    acceleration_limit: float = field(default=1.0, kw_only=True)  # Joint acceleration in m/s^2.


@dataclass
class ControllerConfig:
    name: ControllerType
    gravity_compensation: bool = True
    frame: str = "ee_site"
    pd_gains: FortePDGain | dict[str, PDGain] | None = None

    def pd_gain(self, joint_names: tuple[str, ...]) -> tuple[list[float], list[float]]:
        """Return Kp and Kd arrays in controlled-joint order."""
        if isinstance(self.pd_gains, dict):
            gains = [self.pd_gains[joint] for joint in joint_names]
        else:
            gains = [getattr(self.pd_gains, joint) for joint in joint_names]
        return [gain.kp for gain in gains], [gain.kd for gain in gains]


@dataclass
class RobotConfig:
    controller: ControllerConfig
    constraints: Constraints | None = None
    gripper: GripperConfig | None = None
    model_info: RobotModelInfo = RobotModelInfo()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load YAML fields into this dataclass and its nested configuration dataclasses."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return from_dict(data_class=cls, data=data)

    def update_model_info(self, spec: mujoco.MjSpec) -> None:
        """Refresh local asset names before the spec is attached to a scene."""
        roots = spec.worldbody.bodies
        if len(roots) != 1:
            raise ValueError("Robot assets must have one fixed root body")
        info = RobotModelInfo(
            joint_names=tuple(joint.name for joint in spec.joints),
            actuator_names=tuple(actuator.name for actuator in spec.actuators),
            site_names=tuple(
                site.name for site in spec.sites if site.name and site.parent is not spec.worldbody
            ),
            root_name=roots[0].name,
        )
        if any(not name for name in info.joint_names + info.actuator_names):
            raise ValueError("Robot assets must name their joints and actuators")
        self.model_info = info
