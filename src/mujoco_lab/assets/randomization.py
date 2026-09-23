"""Pure XY sampling for cube placement using supplied footprint half extents."""

import numpy as np


def sample_cube_positions(
    home_xy: np.ndarray,
    half_sizes: np.ndarray,
    rng: np.random.Generator,
    xy_range: float,
    min_gap: float,
    max_attempts: int,
) -> list[np.ndarray]:
    """Sample all cube centers before a caller applies any positions."""
    positions: list[np.ndarray] = []
    for index, home in enumerate(home_xy):
        for _ in range(max_attempts):
            candidate = rng.uniform(home - xy_range, home + xy_range)
            if all(
                np.any(
                    np.abs(candidate - accepted)
                    >= half_sizes[index] + half_sizes[other] + min_gap
                )
                for other, accepted in enumerate(positions)
            ):
                positions.append(candidate)
                break
        else:
            raise ValueError(
                f"Could not place cube{index} without overlap after {max_attempts} attempts"
            )
    return positions
