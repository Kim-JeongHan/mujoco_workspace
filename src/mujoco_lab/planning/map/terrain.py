"""Terrain surfaces and planning-space helpers for path planning examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..space import PlanningSpace

GRID_SIZE = 90
WORLD_SIZE = 14.0
PLAN_SEED = 42
TERRAIN_EDGE_SAMPLE_STEP = 0.18

TerrainPeak = tuple[float, float, float, float]
DEFAULT_PEAKS: tuple[TerrainPeak, ...] = (
    (-4.1, -3.6, 3.4, 1.05),
    (3.7, -3.2, 3.0, 0.5),
    (-2.0, 1.0, 2.8, 1.05),
    (3.4, 3.2, 3.3, 1.20),
)


@dataclass(frozen=True)
class TerrainPlan:
    """Planned path data projected onto a terrain surface."""

    start: np.ndarray
    goal: np.ndarray
    path: np.ndarray
    path_edge_states: list[np.ndarray]
    sampled_nodes: np.ndarray
    graph_edge_states: list[np.ndarray]
    path_length: float


class MountainTerrain:
    """Height-map terrain with a surface metric for path costs."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        world_size: float = WORLD_SIZE,
        peaks: tuple[TerrainPeak, ...] = DEFAULT_PEAKS,
        ridge_height: float = 0.22,
        noise_amplitude: float = 0.12,
    ) -> None:
        """Initialize the terrain height map.

        Args:
            grid_size: Number of samples along each terrain axis
            world_size: Width and height of the square terrain
            peaks: Gaussian peaks as ``(x, y, height, sigma)``
            ridge_height: Amplitude of the broad ridge term
            noise_amplitude: Amplitude of the sinusoidal surface term
        """
        self.grid_size = grid_size
        self.world_size = world_size
        self.peaks = peaks
        self.ridge_height = ridge_height
        self.noise_amplitude = noise_amplitude

        self.xx, self.yy = self._create_grid()
        self.zz = self._create_height_map()
        self.vertices = self._create_vertices()
        self.faces = self._create_faces()
        self._boole_nodes = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        self._boole_weights = np.array([7.0, 32.0, 12.0, 32.0, 7.0], dtype=float) / 90.0

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Return 2D planner bounds for the terrain domain."""
        half_size = self.world_size / 2.0
        return [(-half_size, half_size), (-half_size, half_size)]

    def metric_g(self, x: float, y: float) -> np.ndarray:
        """Return the ambient Euclidean metric at a surface coordinate."""
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

    def metric_h(self, x: float, y: float) -> np.ndarray:
        """Return the induced surface metric at a terrain coordinate."""
        eps = self.world_size / (self.grid_size - 1)

        zx = (self.get_height(x + eps, y) - self.get_height(x - eps, y)) / (2.0 * eps)
        zy = (self.get_height(x, y + eps) - self.get_height(x, y - eps)) / (2.0 * eps)

        return np.array(
            [
                [1.0 + zx * zx, zx * zy],
                [zx * zy, 1.0 + zy * zy],
            ],
            dtype=float,
        )

    def line_cost(
        self,
        start: np.ndarray | tuple[float, float] | list[float],
        goal: np.ndarray | tuple[float, float] | list[float],
    ) -> float:
        """Approximate surface distance between two 2D states."""
        start_state = np.array(start, dtype=float)
        goal_state = np.array(goal, dtype=float)
        direction = goal_state - start_state
        if np.linalg.norm(direction) == 0.0:
            return 0.0

        total = 0.0
        for t, weight in zip(self._boole_nodes, self._boole_weights, strict=True):
            point = start_state + t * direction
            metric = self.metric_h(float(point[0]), float(point[1]))
            integrand = np.sqrt(float(direction.T @ metric @ direction))
            total += float(weight) * integrand
        return float(total)

    def sample_line_states(
        self,
        start: np.ndarray | tuple[float, float] | list[float],
        goal: np.ndarray | tuple[float, float] | list[float],
        max_step: float,
    ) -> np.ndarray:
        """Return dense 2D states along a local chart line."""
        if max_step <= 0:
            raise ValueError("max_step must be positive")

        start_state = np.array(start, dtype=float)
        goal_state = np.array(goal, dtype=float)
        distance = float(np.linalg.norm(goal_state - start_state))
        if distance == 0.0:
            return start_state.reshape(1, -1)

        num_segments = max(1, int(np.ceil(distance / max_step)))
        ts = np.linspace(0.0, 1.0, num_segments + 1)
        return start_state + ts[:, None] * (goal_state - start_state)

    def get_height(self, x: float, y: float) -> float:
        """Return bilinearly interpolated terrain height."""
        half_size = self.world_size / 2.0
        grid_x = (x + half_size) / self.world_size * (self.grid_size - 1)
        grid_y = (y + half_size) / self.world_size * (self.grid_size - 1)

        col0 = int(np.clip(np.floor(grid_x), 0, self.grid_size - 1))
        row0 = int(np.clip(np.floor(grid_y), 0, self.grid_size - 1))
        col1 = min(col0 + 1, self.grid_size - 1)
        row1 = min(row0 + 1, self.grid_size - 1)

        tx = float(grid_x - col0)
        ty = float(grid_y - row0)

        z00 = self.zz[row0, col0]
        z10 = self.zz[row0, col1]
        z01 = self.zz[row1, col0]
        z11 = self.zz[row1, col1]
        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        return float((1.0 - ty) * z0 + ty * z1)

    def states_to_surface_points(
        self,
        states: np.ndarray,
        z_offset: float = 0.12,
    ) -> np.ndarray:
        """Project 2D states to 3D points on the terrain surface."""
        if len(states) == 0:
            return np.empty((0, 3), dtype=float)

        points = np.empty((len(states), 3), dtype=float)
        points[:, :2] = states[:, :2]
        for index, (x, y) in enumerate(states[:, :2]):
            points[index, 2] = self.get_height(float(x), float(y)) + z_offset
        return points

    def _create_grid(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(-self.world_size / 2.0, self.world_size / 2.0, self.grid_size)
        ys = np.linspace(-self.world_size / 2.0, self.world_size / 2.0, self.grid_size)
        xx, yy = np.meshgrid(xs, ys)
        return xx, yy

    def _create_height_map(self) -> np.ndarray:
        zz = np.zeros_like(self.xx)
        for x, y, height, sigma in self.peaks:
            distance_squared = (self.xx - x) ** 2 + (self.yy - y) ** 2
            zz += height * np.exp(-distance_squared / (2.0 * sigma**2))

        ridge = self.ridge_height * np.exp(-((self.yy + 0.25 * self.xx) ** 2) / 2.2)
        surface_noise = self.noise_amplitude * np.sin(2.2 * self.xx) * np.cos(1.7 * self.yy)
        zz += ridge + surface_noise
        zz -= zz.min()
        return zz

    def _create_vertices(self) -> np.ndarray:
        return np.stack([self.xx.ravel(), self.yy.ravel(), self.zz.ravel()], axis=1)

    def _create_faces(self) -> np.ndarray:
        faces = []
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size - 1):
                i = row * self.grid_size + col
                faces.append([i, i + 1, i + self.grid_size])
                faces.append([i + 1, i + self.grid_size + 1, i + self.grid_size])
        return np.array(faces, dtype=np.uint32)


class TerrainRiemannianSpace(PlanningSpace):
    """Planning space whose edge costs follow the terrain surface metric."""

    def __init__(self, terrain: MountainTerrain) -> None:
        """Initialize the planning space for terrain-surface edges."""
        self.terrain = terrain

    def distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Return terrain surface distance between states."""
        return self.terrain.line_cost(start, goal)

    def edge_states(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return dense edge states used for collision checks and rendering."""
        return self.terrain.sample_line_states(
            start,
            goal,
            max_step=TERRAIN_EDGE_SAMPLE_STEP,
        )

    def edge_cost(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Return Riemannian Line-R cost for the terrain-chart edge."""
        return self.distance(start, goal)


def create_random_start_goal(
    bounds: list[tuple[float, float]],
    seed: int,
    minimum_distance: float = WORLD_SIZE * 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic random start/goal pair within bounds."""
    rng = np.random.default_rng(seed)
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)

    for _ in range(100):
        start = rng.uniform(lower, upper)
        goal = rng.uniform(lower, upper)
        if float(np.linalg.norm(goal - start)) >= minimum_distance:
            return start, goal

    return lower, upper
