"""Typed robot controller settings loaded from robot.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians
from pathlib import Path
from typing import Literal, Self

import mujoco
import yaml
from dacite import Config, from_dict

# dacite requires a runtime Literal alias rather than a PEP 695 alias.
ControllerType = Literal["none", "pd", "position", "osc"]  # noqa: UP040


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
class PoseConfig:
    """Default and optional zero arm poses, with gripper opening width in meters.

    Robots without a configured gripper contain only arm joint positions.
    Arm values follow asset joint order, excluding the configured finger joints.
    Direct construction uses radians; from_yaml converts YAML arm angles from degrees.
    """

    default: list[float]
    zero: list[float] | None = None

    @classmethod
    def from_yaml(cls, data: dict[str, list[float] | None], *, has_gripper: bool) -> Self:
        """Read default and optional zero poses, converting only arm angles to radians."""
        pose = from_dict(data_class=cls, data=data, config=Config(strict=True))

        def convert(values: list[float]) -> list[float]:
            arm_count = len(values) - int(has_gripper)
            return [*[radians(value) for value in values[:arm_count]], *values[arm_count:]]

        return cls(
            default=convert(pose.default),
            zero=None if pose.zero is None else convert(pose.zero),
        )

    def named_poses(self) -> dict[str, list[float]]:
        """Return the default pose and the zero pose when configured."""
        poses = {"default": self.default}
        if self.zero is not None:
            poses["zero"] = self.zero
        return poses


@dataclass
class GripperConfig:
    actuator: str
    joints: list[str]  # Zero-based slide fingers sharing a 1:1 position target.
    velocity_limit: float = field(default=0.4, kw_only=True)  # Opening-width speed in m/s.
    acceleration_limit: float = field(default=2.0, kw_only=True)  # Width acceleration in m/s^2.


@dataclass
class ControllerConfig:
    name: ControllerType
    gravity_compensation: bool = True
    frame: str = "grasp"
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
    pose: PoseConfig = field(kw_only=True)
    model_info: RobotModelInfo = RobotModelInfo()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load YAML settings, converting arm poses and angular bounds to radians."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        data["pose"] = PoseConfig.from_yaml(
            data["pose"], has_gripper=data.get("gripper") is not None
        )
        config = from_dict(data_class=cls, data=data)
        if config.constraints is not None:
            config.constraints.position_limit = [
                [radians(value) for value in bounds] for bounds in config.constraints.position_limit
            ]
        return config

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
