"""Convert Forte arm coordinates to the forward-extended zero reference.

The seven entries follow shoulder yaw/pitch/roll, elbow pitch, forearm roll,
wrist pitch, and wrist roll. Constant reference shifts do not affect joint
velocities or accelerations. These are simulation references, not calibrated
hardware encoder offsets.
"""

import numpy as np

ZERO_LEGACY_DEGREES = np.array(
    [
        0.0,
        90.0,
        -7.401153250875606,
        94.24000009617235,
        8.092777269547748,
        -73.28564411159464,
        8.371176193898398,
    ]
)
ZERO_LEGACY_RADIANS = np.deg2rad(ZERO_LEGACY_DEGREES)
HOME_DEGREES = np.array([0.0, -90.0, 0.0, -90.0, 0.0, 90.0, 0.0])
HOME_RADIANS = np.deg2rad(HOME_DEGREES)


def from_legacy(positions):
    """Shift one arm pose or a batch of poses, preserving physical geometry."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 0 or positions.shape[-1] != 7:
        raise ValueError("Forte arm positions must have seven coordinates on the last axis")
    return positions - ZERO_LEGACY_RADIANS
