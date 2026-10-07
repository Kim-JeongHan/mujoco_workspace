"""Pure XY and yaw sampling for cube placement with footprint spacing."""

import numpy as np


def sample_cube_positions(
    home_xy: np.ndarray,
    half_sizes: np.ndarray,
    rng: np.random.Generator,
    xy_range: float,
    min_gap: float,
    max_attempts: int,
    *,
    yaw_range_degrees: float = 0.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Sample centers and yaw offsets in degrees before applying any cube poses.

    Half sizes describe initially axis-aligned footprints. Rotated footprints
    use axis-aligned bounding boxes to keep at least ``min_gap`` clearance.
    Zero yaw range preserves the existing XY random-number sequence.
    """
    yaws = np.zeros(len(home_xy))
    if yaw_range_degrees:
        yaws = rng.uniform(-yaw_range_degrees, yaw_range_degrees, size=len(home_xy))
        angles = np.deg2rad(yaws)
        cosines = np.abs(np.cos(angles))
        sines = np.abs(np.sin(angles))
        half_sizes = np.column_stack(
            (
                cosines * half_sizes[:, 0] + sines * half_sizes[:, 1],
                sines * half_sizes[:, 0] + cosines * half_sizes[:, 1],
            )
        )
    positions: list[np.ndarray] = []
    for index, home in enumerate(home_xy):
        for _ in range(max_attempts):
            candidate = rng.uniform(home - xy_range, home + xy_range)
            if all(
                np.any(
                    np.abs(candidate - accepted) >= half_sizes[index] + half_sizes[other] + min_gap
                )
                for other, accepted in enumerate(positions)
            ):
                positions.append(candidate)
                break
        else:
            raise ValueError(
                f"Could not place cube{index} without overlap after {max_attempts} attempts"
            )
    return positions, yaws
