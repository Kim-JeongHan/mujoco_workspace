"""Sampling strategies for RRT algorithms."""

import numpy as np


class Sampler:
    """Base class for state space samplers."""

    def __init__(self, bounds: list[tuple[float, float]], seed: int | None = None) -> None:
        """Initialize the sampler.

        Args:
            bounds: List of (min, max) tuples for each dimension
            seed: Random seed for reproducibility
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)

        if seed is not None:
            np.random.seed(seed)

    def sample(self) -> np.ndarray:
        """Sample a random state from the state space.

        Returns:
            A random state vector
        """
        raise NotImplementedError


class UniformSampler(Sampler):
    """Uniform random sampler."""

    def sample(self) -> np.ndarray:
        """Sample uniformly from the state space.

        Returns:
            A uniformly sampled state vector
        """
        result: np.ndarray = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        return result


class GoalBiasedSampler(Sampler):
    """Goal-biased sampler that samples the goal with some probability."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        goal_state: np.ndarray,
        goal_bias: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """Initialize the goal-biased sampler.

        Args:
            bounds: List of (min, max) tuples for each dimension
            goal_state: The goal state to bias towards
            goal_bias: Probability of sampling the goal (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        super().__init__(bounds, seed)
        self.goal_state = np.array(goal_state)
        self.goal_bias = goal_bias

        if len(self.goal_state) != self.dim:
            raise ValueError(
                f"Goal state dimension {len(self.goal_state)} does not match bounds dimension {self.dim}"
            )

    def sample(self) -> np.ndarray:
        """Sample with goal bias.

        Returns:
            Either the goal state or a uniformly sampled state
        """
        if np.random.random() < self.goal_bias:
            return self.goal_state.copy()
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])  # type: ignore[no-any-return]


class InformedSampler(Sampler):
    """Informed sampler that samples within an ellipsoidal region."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        start_state: np.ndarray,
        goal_state: np.ndarray,
        seed: int | None = None,
    ) -> None:
        """Initialize the informed sampler.

        Args:
            bounds: List of (min, max) tuples for each dimension
            start_state: The start state
            goal_state: The goal state
            seed: Random seed for reproducibility
        """
        super().__init__(bounds, seed)
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        self.c_min = np.linalg.norm(self.goal_state - self.start_state)
        self.x_center = (self.start_state + self.goal_state) / 2.0
        self.dim = len(bounds)
        self.R = self._compute_ellipsoid_rotation_matrix(
            self.dim, self.start_state, self.goal_state
        )

        if len(self.start_state) != self.dim or len(self.goal_state) != self.dim:
            raise ValueError("Start or goal state dimension does not match bounds dimension")

    def sample(self, c_best: float | None = None) -> np.ndarray:
        """Sample within the informed ellipsoid.

        Args:
            c_best: Current best path cost. If None or infinite, falls back to uniform sampling.

        Returns:
            A sampled state vector within the ellipsoid
        """
        if c_best is not None and np.isfinite(c_best):
            return self._sample_informed_ellipsoid(max(float(c_best), float(self.c_min)))
        else:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])  # type: ignore[no-any-return]

    def _sample_informed_ellipsoid(self, c_best: float) -> np.ndarray:
        """Cholesky-based Informed sampling within the ellipsoid

        Samples uniformly inside the prolate hyperspheroid (Informed set):
            (x - x_center)^T * S * (x - x_center) <= 1
            S = L^T * L
            x_ellipsoid = x_center + L * x_ball

            reference :  (https://arxiv.org/abs/1404.1347)

        Where:
            x_ball: Sampled point in unit n-ball
            L: Scaling matrix
            C: Rotation matrix

        Args:
            c_best: Current best path cost
        Returns:
            A sampled state vector within the ellipsoid
        """
        while True:
            minor_axis = np.sqrt(max(float(c_best**2 - self.c_min**2), 0.0))
            scale_matrix = np.diag([c_best / 2.0] + [minor_axis / 2.0] * (self.dim - 1))

            ball_sample = np.random.normal(0, 1, self.dim)
            sample_norm = np.linalg.norm(ball_sample)
            if sample_norm == 0:
                continue
            ball_sample /= sample_norm
            r = np.random.uniform(0, 1) ** (1 / self.dim)
            ball_sample *= r
            sample = self.x_center + self.R @ scale_matrix @ ball_sample
            if np.all(sample >= self.bounds[:, 0]) and np.all(sample <= self.bounds[:, 1]):
                return sample

    def _compute_ellipsoid_rotation_matrix(
        self, dim: int, start: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Compute the rotation matrix to align the ellipsoid with the start-goal line.

        Args:
            dim: Dimension of the state space
            start: Start state vector
            goal: Goal state vector

        Returns:
            Rotation matrix
        """
        if self.c_min == 0:
            return np.eye(dim)

        a1 = (goal - start) / self.c_min
        i1 = np.zeros((dim, dim))
        i1[:, 0] = a1
        u, _, vt = np.linalg.svd(i1)
        m = u @ vt
        if np.linalg.det(m) < 0:
            u[:, -1] *= -1
            m = u @ vt
        return m
