"""Reference posture and minimum-jerk interpolation from the Forte demos."""

import numpy as np

HOME_QPOS = np.array([0.0, -1.0472, 0.0, 1.74533, 0.0, 0.0, 0.0])


def min_jerk(fraction: float) -> tuple[float, float]:
    """Minimum-jerk blend and its derivative."""
    u = float(np.clip(fraction, 0.0, 1.0))
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    slope = 30 * u**2 - 60 * u**3 + 30 * u**4
    return blend, slope
