"""Robot assets, state/controller composition, and scoped actuator inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_lab.assets import ROBOT_SCENES
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.state import RobotState
from mujoco_lab.utils import Transform

if TYPE_CHECKING:
    from mujoco_lab.control.base import Controller
    from mujoco_lab.simulation import Simulator


@dataclass(frozen=True)
class RobotSpec:
    """Place a named asset in world coordinates; a singleton may use the mount."""

    name: str
    robot_type: str
    pose: Transform | None = None


@dataclass(frozen=True)
class AssetInfo:
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    site_names: tuple[str, ...]
    root_name: str


def _asset_info(spec: mujoco.MjSpec) -> AssetInfo:
    roots = spec.worldbody.bodies
    if len(roots) != 1:
        raise ValueError("Robot assets must have one fixed root body")
    return AssetInfo(
        tuple(joint.name for joint in spec.joints),
        tuple(actuator.name for actuator in spec.actuators),
        tuple(site.name for site in spec.sites if site.name and site.parent is not spec.worldbody),
        roots[0].name,
    )


def _load_asset(name: str):
    path = ROBOT_SCENES[name]

    spec = mujoco.MjSpec.from_file(str(path))
    if (
        any(
            actuator.dyntype != mujoco.mjtDyn.mjDYN_NONE or actuator.actdim > 0
            for actuator in spec.actuators
        )
        or any(body.mocap for body in spec.bodies)
        or any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in spec.joints)
    ):
        raise ValueError("Robot assets must be fixed-base with no activation or mocap state")
    info = _asset_info(spec)
    if any(not name for name in info.joint_names + info.actuator_names):
        raise ValueError("Robot assets must name their joints and actuators")
    return spec, info


class Robot:
    """Connect one robot's state and controller to its actuator inputs.

    RobotState owns state reads and dynamics. Simulator owns physics advancement.
    """

    def __init__(
        self,
        simulator: Simulator,
        info: AssetInfo,
        prefix: str,
        robot_type: str,
    ) -> None:
        self._simulator = simulator
        self.model, self.data = simulator.model, simulator.data
        self.name = prefix.removesuffix("/")
        self.robot_type = robot_type
        self.prefix = prefix
        self.state = RobotState(
            self.model,
            self.data,
            name=self.name,
            prefix=prefix,
            root_name=info.root_name,
            joint_names=info.joint_names,
            site_names=info.site_names,
        )
        self.actuator_names = info.actuator_names
        self.actuator_ids = [self.model.actuator(prefix + name).id for name in info.actuator_names]
        self.nu = len(self.actuator_ids)
        self.controller: Controller | None = None
        self.joint_state = self.state.snapshot()
        self.target: ControlTarget | None = None

    def _apply_home_keyframe(self) -> None:
        """Apply this robot's home state without forwarding the shared scene."""
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.prefix + "home")
        if home < 0:
            return
        key = self.model.key(home)
        self.data.qpos[self.state.qpos_indices] = key.qpos[self.state.qpos_indices]
        self.data.qvel[self.state.dof_indices] = key.qvel[self.state.dof_indices]
        self.data.ctrl[self.actuator_ids] = key.ctrl[self.actuator_ids]

    def change_controller(self, controller: Controller | None) -> None:
        """Bind and reset a replacement controller, or detach it with None."""
        self._simulator._require_idle()
        if controller is self.controller:
            return
        if controller is not None:
            controller.bind(self.state)
            controller.reset()
            self.update_state()
        self.controller = controller
        self.target = None if controller is None else controller.initial_target(self.joint_state)

    def update_state(self) -> None:
        """Refresh and return the owned joint snapshot without forwarding or stepping."""
        self.joint_state = self.state.snapshot()

    def control(self) -> bool:
        """Apply only this robot's native actuator commands; return saturation status.

        Control inputs use each actuator's native units. Only actuators with
        enabled control limits are clipped. This method never steps physics.
        """
        values = self.controller.compute(self.joint_state, self.target)
        command = np.asarray(values, dtype=float)
        if command.shape != (self.nu,) or not np.isfinite(command).all():
            raise ValueError(f"Robot {self.name!r} needs {self.nu} finite actuator inputs")
        limits = self.model.actuator_ctrlrange[self.actuator_ids]
        limited = self.model.actuator_ctrllimited[self.actuator_ids]
        clipped = np.where(limited, np.clip(command, limits[:, 0], limits[:, 1]), command)

        # update control information into mujoco
        self.data.ctrl[self.actuator_ids] = clipped
        return bool(np.any(clipped != command))

    def get_tracking_error(self):
        return self.controller.tracking_error
