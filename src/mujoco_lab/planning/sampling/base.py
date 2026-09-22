"""Base class for all sampling algorithms."""

from abc import ABC, abstractmethod

import numpy as np

from ..collision import CollisionChecker, EmptyCollisionChecker, ObstacleCollisionChecker
from ..graph import Graph, Node
from ..search import AStar


class RRTBase(ABC):
    """Base class for RRT algorithms."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        max_iterations: int = 5000,
        step_size: float = 0.5,
        goal_tolerance: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the RRT base planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            max_iterations: Maximum number of iterations
            step_size: Maximum distance to extend in each iteration
            goal_tolerance: Distance threshold to consider goal reached
            seed: Random seed for reproducibility
        """
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        self.dim = len(start_state)
        if len(self.start_state) != len(goal_state):
            raise ValueError("Start and goal states must have the same dimension")

        self.bounds = bounds
        if len(self.bounds) != self.dim:
            raise ValueError("Bounds must have the same dimension as the start and goal states")

        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.seed = seed

        # Collision checker
        self.collision_checker: CollisionChecker | ObstacleCollisionChecker | EmptyCollisionChecker
        if collision_checker is None:
            self.collision_checker = EmptyCollisionChecker()
        else:
            self.collision_checker = collision_checker

        self.root: Node | None = None
        self.goal_node: Node | None = None
        self.graph = Graph()

    def _check_start_goal_collision(self) -> bool:
        """Check if start and goal states are collision-free.

        Returns:
            True if both are collision-free, False otherwise
        """
        if not self.collision_checker.is_collision_free(self.start_state):
            print("Start state is in collision!")
            return False

        if not self.collision_checker.is_collision_free(self.goal_state):
            print("Goal state is in collision!")
            return False

        return True

    def _connect_goal(self, node: Node) -> Node | None:
        """Return an exact-goal node when node can validly finish the path."""
        goal_node = Node(state=self.goal_state)
        if self.graph.distance(node, goal_node) > self.goal_tolerance:
            return None
        if not self.graph.is_edge_collision_free(node, goal_node, self.collision_checker):
            return None
        if np.allclose(node.state, self.goal_state):
            return node

        goal_node.change_parent(node, node.cost + self.graph.edge_cost(node, goal_node))
        return goal_node

    @abstractmethod
    def plan(self) -> list[Node] | None:
        """Run the planning algorithm.

        Returns:
            List of nodes from start to goal, or None if no path found
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get statistics about the planning process.

        Returns:
            Dictionary with statistics
        """
        pass

    @abstractmethod
    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the tree(s) for visualization.

        Returns:
            List of all nodes explored during planning
        """
        pass

    @abstractmethod
    def get_goal_node(self) -> Node | None:
        """Get the goal node (or connection point) for visualization.

        Returns:
            The node that reached the goal, or None if planning failed
        """
        pass


class RRGBase(RRTBase):
    """Base class for RRG algorithm."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        max_iterations: int = 1000,
        step_size: float = 0.5,
        goal_tolerance: float = 0.5,
        radius_gain: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the RRG base class.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance
            max_iterations: Maximum number of iterations to run
            step_size: Maximum distance to extend tree at each iteration
            goal_tolerance: Distance threshold to consider goal reached
            radius_gain: Scaling factor for connection radius
            seed: Random seed for reproducibility
        """
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=max_iterations,
            step_size=step_size,
            goal_tolerance=goal_tolerance,
            seed=seed,
        )

        self.path: list[Node] | None = None

        # A*
        self.astar = AStar(self.graph)

        # Radius gain
        self.radius_gain = radius_gain

    def get_near_node(self, target: Node) -> list[Node]:
        """Get the near nodes of the target node."""
        num_nodes = len(self.graph.nodes)

        if num_nodes <= 1:
            return []

        return self.graph.near(target, self._connection_radius())

    def _connection_radius(self) -> float:
        """Return the connection radius used by near-node queries."""
        num_nodes = len(self.graph.nodes)
        return float(self.radius_gain * np.power(np.log(num_nodes) / num_nodes, 1 / self.dim))

    def get_path_length(self) -> float:
        """Get the total length of the current path."""
        if self.path is None:
            return float("inf")

        if len(self.path) < 2:
            return 0.0
        distances = [
            self.graph.distance(self.path[i], self.path[i + 1]) for i in range(len(self.path) - 1)
        ]
        return float(np.sum(distances)) if distances else float("inf")
