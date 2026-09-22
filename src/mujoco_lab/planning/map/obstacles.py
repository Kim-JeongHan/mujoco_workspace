"""Obstacle generation and management module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import viser


class ObstacleType(Enum):
    """Enum for obstacle types."""

    BOX = "box"
    SPHERE = "sphere"


class Obstacle(ABC):
    """Abstract base class for obstacles."""

    @abstractmethod
    def get_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Returns the bounds of the obstacle.

        Returns:
            ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        pass

    @abstractmethod
    def contains_point(self, point: tuple[float, float, float]) -> bool:
        """Check if a point is inside the obstacle.

        Args:
            point: 3D point to check (x, y, z)

        Returns:
            True if point is inside the obstacle
        """
        pass

    @abstractmethod
    def intersects(self, other: Obstacle) -> bool:
        """Check if this obstacle intersects with another obstacle.

        Args:
            other: Another obstacle to check intersection with

        Returns:
            True if obstacles intersect
        """
        pass


class BoxObstacle(Obstacle):
    """Box-shaped obstacle (AABB - Axis-Aligned Bounding Box)."""

    def __init__(
        self, position: tuple[float, float, float], size: tuple[float, float, float]
    ) -> None:
        """Initialize a box obstacle.

        Args:
            position: Center position of the obstacle (x, y, z)
            size: Size of the obstacle (width, depth, height)
        """
        self.position = position
        self.size = size

    def get_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Returns the bounds of the box obstacle.

        Returns:
            ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        x, y, z = self.position
        w, d, h = self.size

        return ((x - w / 2, x + w / 2), (y - d / 2, y + d / 2), (z - h / 2, z + h / 2))

    def contains_point(self, point: tuple[float, float, float]) -> bool:
        """Check if a point is inside the box obstacle.

        Args:
            point: 3D point to check (x, y, z)

        Returns:
            True if point is inside the box
        """
        x, y, z = point
        bounds = self.get_bounds()

        return (
            bounds[0][0] <= x <= bounds[0][1]
            and bounds[1][0] <= y <= bounds[1][1]
            and bounds[2][0] <= z <= bounds[2][1]
        )

    def intersects(self, other: Obstacle) -> bool:
        """Check if this box intersects with another obstacle using AABB collision detection.

        Args:
            other: Another obstacle to check intersection with

        Returns:
            True if obstacles intersect
        """
        self_bounds = self.get_bounds()
        other_bounds = other.get_bounds()

        # AABB (Axis-Aligned Bounding Box) collision detection
        for i in range(3):
            if self_bounds[i][1] < other_bounds[i][0] or self_bounds[i][0] > other_bounds[i][1]:
                return False
        return True


class SphereObstacle(Obstacle):
    """Sphere-shaped obstacle."""

    def __init__(self, center: tuple[float, float, float], radius: float) -> None:
        """Initialize a sphere obstacle.

        Args:
            center: Center position of the sphere (x, y, z)
            radius: Radius of the sphere
        """
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def get_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Returns the bounding box of the sphere.

        Returns:
            ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        x, y, z = self.center
        r = self.radius

        return ((x - r, x + r), (y - r, y + r), (z - r, z + r))

    def contains_point(self, point: tuple[float, float, float]) -> bool:
        """Check if a point is inside the sphere.

        Args:
            point: 3D point to check (x, y, z)

        Returns:
            True if point is inside the sphere
        """
        distance = np.linalg.norm(np.array(point) - self.center)
        return float(distance) <= self.radius

    def intersects(self, other: Obstacle) -> bool:
        """Check if this sphere intersects with another obstacle.

        Args:
            other: Another obstacle to check intersection with

        Returns:
            True if obstacles intersect
        """
        if isinstance(other, SphereObstacle):
            # Sphere-to-sphere: check if distance between centers <= sum of radii
            distance = np.linalg.norm(self.center - other.center)
            return float(distance) <= (self.radius + other.radius)
        else:
            # Sphere-to-box: use AABB approximation
            # Find closest point on the other obstacle's bounding box to sphere center
            other_bounds = other.get_bounds()
            closest_point = []

            for i in range(3):
                closest_point.append(
                    max(other_bounds[i][0], min(self.center[i], other_bounds[i][1]))
                )

            # Check if closest point is within sphere
            distance = np.linalg.norm(np.array(closest_point) - self.center)
            return float(distance) <= self.radius


def _generate_random_box_obstacle(
    map_size: float,
    min_size: float,
    max_size: float,
    min_height: float,
    max_height: float,
) -> BoxObstacle:
    """Generate a random box obstacle.

    Args:
        map_size: Size of the map
        min_size: Minimum obstacle size
        max_size: Maximum obstacle size
        min_height: Minimum height
        max_height: Maximum height

    Returns:
        Random BoxObstacle instance
    """
    # Generate random size
    width = np.random.uniform(min_size, max_size)
    depth = np.random.uniform(min_size, max_size)
    height = np.random.uniform(min_height, max_height)

    # Generate random position (within map boundaries)
    x = np.random.uniform(-map_size / 2 + width / 2, map_size / 2 - width / 2)
    y = np.random.uniform(-map_size / 2 + depth / 2, map_size / 2 - depth / 2)
    z = height / 2  # Start from ground

    return BoxObstacle(position=(x, y, z), size=(width, depth, height))


def _generate_random_sphere_obstacle(
    map_size: float,
    min_size: float,
    max_size: float,
    min_height: float,
    max_height: float,
) -> SphereObstacle:
    """Generate a random sphere obstacle.

    Args:
        map_size: Size of the map
        min_size: Minimum radius
        max_size: Maximum radius
        min_height: Minimum height for center (z position)
        max_height: Maximum height for center (z position)

    Returns:
        Random SphereObstacle instance
    """
    # Generate random radius
    radius = np.random.uniform(min_size, max_size)

    # Generate random position (within map boundaries)
    x = np.random.uniform(-map_size / 2 + radius, map_size / 2 - radius)
    y = np.random.uniform(-map_size / 2 + radius, map_size / 2 - radius)
    z = np.random.uniform(min_height + radius, max_height)  # Ensure sphere doesn't go below ground

    return SphereObstacle(center=(x, y, z), radius=radius)


def _visualize_obstacle(
    server: viser.ViserServer,
    obstacle: Obstacle,
    index: int,
    color: tuple[int, int, int],
) -> None:
    """Visualize an obstacle in Viser.

    Args:
        server: Viser server instance
        obstacle: Obstacle to visualize
        index: Index for naming
        color: Obstacle color
    """
    if isinstance(obstacle, BoxObstacle):
        width, depth, height = obstacle.size
        x, y, z = obstacle.position
        server.scene.add_box(
            f"/obstacles/box_{index}",
            dimensions=(width, depth, height),
            position=(x, y, z),
            color=color,
        )
    elif isinstance(obstacle, SphereObstacle):
        x, y, z = obstacle.center
        server.scene.add_icosphere(
            f"/obstacles/sphere_{index}",
            radius=obstacle.radius,
            position=(x, y, z),
            color=color,
        )
