"""Original Forte joint-space PD equations, gains, and waypoint routine."""

import numpy as np

from mujoco_lab.control.base import Controller
from mujoco_lab.control.trajectory import HOME_QPOS, min_jerk
from mujoco_lab.state import JointState

# shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, lower_arm_roll,
# wrist_pitch, wrist_roll
ROUTINE = np.array(
    [
        HOME_QPOS,
        [0.0, -1.40, 0.0, 1.20, 0.0, 0.30, 0.0],
        [0.9, -1.10, 0.0, 1.50, 0.0, 0.20, 1.2],
        [-0.9, -1.10, 0.0, 1.50, 0.0, 0.20, -1.2],
        [0.0, -0.80, 0.0, 0.60, 0.0, 0.20, 0.0],
        [0.0, -2.00, 0.0, 2.40, 0.0, 0.60, 0.0],
    ]
)

# Per-joint PD gains. Sized from the inertia diagonal so the wrist is not
# over-gained relative to the shoulder.
KP = np.array([147.0, 222.0, 47.0, 74.0, 6.8, 7.2, 3.2])
KD = np.array([11.8, 17.8, 3.8, 5.9, 0.55, 0.57, 0.25])


class JointSpacePD(Controller):
    name = "joint-space PD"

    def __init__(
        self,
        waypoints: np.ndarray = ROUTINE,
        kp: np.ndarray = KP,
        kd: np.ndarray = KD,
        transit: float = 1.5,
        hold: float = 0.4,
        gravity_compensation: bool = True,
    ) -> None:
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.transit = transit
        self.hold = hold
        self.gravity_compensation = gravity_compensation
        self.tracking_error = 0.0
        self._desired = self.waypoints[0].copy()

    def desired(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        segment = self.transit + self.hold
        count = len(self.waypoints)
        phase = elapsed % (count * segment)
        index = int(phase // segment)
        local = phase - index * segment
        start = self.waypoints[index]
        end = self.waypoints[(index + 1) % count]
        if local >= self.transit:
            return end.copy(), np.zeros_like(end)
        blend, slope = min_jerk(local / self.transit)
        delta = end - start
        return start + delta * blend, delta * slope / self.transit

    def torques(self, state: JointState) -> np.ndarray:
        position, velocity = self.desired(state.time)
        self._desired = position
        error = position - state.qpos
        self.tracking_error = float(np.abs(error).max())
        torque = self.kp * error + self.kd * (velocity - state.qvel)
        if self.gravity_compensation:
            torque = torque + state.bias_forces
        return torque

    def summary(self) -> str:
        return f"waypoints: {len(self.waypoints)}, transit: {self.transit}s"
