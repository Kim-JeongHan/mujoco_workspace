"""Compose and run one MuJoCo scene with robot-local state and controllers."""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType

import mujoco
import numpy as np

from mujoco_lab.robot import Robot, RobotSpec, _load_asset
from mujoco_lab.stats import RunStats
from mujoco_lab.utils import StateMachine

__all__ = ["Simulator"]


class SimulatorState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    VIEWING = "viewing"
    RESETTING = "resetting"
    RENDERING = "rendering"


STATE_TRANSITIONS = {
    SimulatorState.IDLE: {
        SimulatorState.RUNNING,
        SimulatorState.VIEWING,
        SimulatorState.RESETTING,
        SimulatorState.RENDERING,
    },
    SimulatorState.RUNNING: {SimulatorState.IDLE},
    SimulatorState.VIEWING: {SimulatorState.IDLE},
    SimulatorState.RESETTING: {SimulatorState.IDLE},
    SimulatorState.RENDERING: {SimulatorState.IDLE},
}


class Simulator:
    """One shared scene with immutable robot bindings and one active stepping owner.

    Headless and GUI stepping use the same control-before-mj_step sequence.
    Headless running covers both control evaluation and physics advancement.
    Lifecycle operations return to idle only after successful work. Neither mode
    is an RL environment batch.
    """

    def __init__(
        self,
        scene: mujoco.MjSpec,
        *,
        robots: list[RobotSpec] | tuple[RobotSpec, ...] = (),
        dt: float = 0.002,
    ) -> None:
        """Compile a scene and capture its fully initialized reset state."""
        names = [item.name for item in robots]
        if any(not name or "/" in name for name in names) or len(set(names)) != len(names):
            raise ValueError("Robot instance names must be unique, nonempty, and contain no '/'")
        if len(robots) > 1 and any(item.pose is None for item in robots):
            raise ValueError("Multiple robots require explicit world poses")

        # Treat the supplied spec as a reusable scene blueprint.
        scene = scene.copy()
        robot_info = []
        for robot_spec in robots:
            asset, info = _load_asset(robot_spec.robot_type)
            prefix = robot_spec.name + "/"
            if robot_spec.pose is None:
                mount = scene.site("robot_mount")
                scene.attach(asset, prefix=prefix, site=mount)
            else:
                pose = robot_spec.pose.as_xyzquat()
                frame = scene.worldbody.add_frame(pos=pose[:3], quat=pose[3:])
                scene.attach(asset, prefix=prefix, frame=frame)
            robot_info.append((robot_spec.name, info, prefix, robot_spec.robot_type))

        model = scene.compile()
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.dt = dt
        model.opt.timestep = dt
        self.model = model
        self.data = mujoco.MjData(model)
        self.robots = MappingProxyType(
            {
                name: Robot(self, info, prefix, robot_type)
                for name, info, prefix, robot_type in robot_info
            }
        )
        self._state = StateMachine(SimulatorState.IDLE, STATE_TRANSITIONS)
        self._stop_requested = False
        self.target_updater = None

        environment_home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if environment_home >= 0:
            mujoco.mj_resetDataKeyframe(model, self.data, environment_home)
        for robot in self.robots.values():
            robot._apply_home_keyframe()
        mujoco.mj_forward(model, self.data)
        for robot in self.robots.values():
            robot.update_state()
        self._initial_state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._initial_state = np.empty(mujoco.mj_stateSize(model, self._initial_state_spec))
        mujoco.mj_getState(model, self.data, self._initial_state, self._initial_state_spec)

    def _require_idle(self):
        if self._state.state is not SimulatorState.IDLE:
            raise RuntimeError(f"Simulator is busy with {self._state.state}")

    def physics_step(self):
        """Evaluate control and advance one physics tick."""
        samples = {}
        for robot in self.robots.values():
            robot.update_state()
        if self.target_updater is not None:
            self.target_updater(self)
        for name, robot in self.robots.items():
            if robot.controller is None:
                continue
            saturated = robot.control()
            samples[name] = (saturated, robot.get_tracking_error())
        for robot in self.robots.values():
            robot.update_state()
        mujoco.mj_step(self.model, self.data)
        return samples

    def step(self):
        """Advance the shared scene once and return fresh per-robot statistics."""
        return self.run_steps(1)

    def run_steps(self, steps: int):
        """Advance physics and control together for each tick."""

        if steps < 0:
            raise ValueError("steps must be zero or greater")
        self._require_idle()
        self._stop_requested = False
        self._state.transition(SimulatorState.RUNNING)
        stats = {name: RunStats() for name in self.robots}
        for _ in range(steps):
            if self._stop_requested:
                break
            samples = self.physics_step()
            for name, result in stats.items():
                result.update(*samples.get(name, (False, None)))
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Simulation diverged")
        self._state.transition(SimulatorState.IDLE)
        self._stop_requested = False
        return stats

    def stop(self) -> None:
        """Request the active run or viewer to stop after its current physics step."""
        if self._state.state in (SimulatorState.RUNNING, SimulatorState.VIEWING):
            self._stop_requested = True

    def reset(self) -> None:
        """Restore the complete initial state and clear controller history, not gains."""

        self._state.transition(SimulatorState.RESETTING)
        self._stop_requested = False
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(
            self.model,
            self.data,
            self._initial_state,
            self._initial_state_spec,
        )
        mujoco.mj_forward(self.model, self.data)
        for robot in self.robots.values():
            if robot.controller is not None:
                robot.controller.reset()
            robot.update_state()
            if robot.controller is None:
                robot.target = None
            else:
                robot.target = robot.controller.initial_target(robot.joint_state)
        self._state.transition(SimulatorState.IDLE)
