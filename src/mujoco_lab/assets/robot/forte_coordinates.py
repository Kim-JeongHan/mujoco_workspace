"""Forte arm reference poses and source-model zero offsets.

The seven entries follow shoulder yaw/pitch/roll, elbow pitch, forearm roll,
wrist pitch, and wrist roll. Constant reference shifts do not affect joint
velocities or accelerations. These are simulation references, not calibrated
hardware encoder offsets.
"""

import numpy as np

SOURCE_ZERO_DEGREES = np.array(
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
SOURCE_ZERO_RADIANS = np.deg2rad(SOURCE_ZERO_DEGREES)
HOME_DEGREES = np.array([0.0, -90.0, 0.0, -90.0, 0.0, 90.0, 0.0])
HOME_RADIANS = np.deg2rad(HOME_DEGREES)
