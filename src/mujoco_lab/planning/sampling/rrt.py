"""RRT (Rapidly-exploring Random Tree) algorithm implementation."""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm import tqdm

from ..collision import CollisionChecker
from ..graph import Graph, Node
from ..space import EuclideanSpace, PlanningSpace
from .base import RRGBase, RRTBase
from .sampler import GoalBiasedSampler, InformedSampler, Sampler, UniformSampler


class RRTConfig(BaseModel):
    """Configuration for RRT algorithm."""

    sampler: type[Sampler] = GoalBiasedSampler
    sampler_kwargs: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = Field(default=5000, ge=0)
    step_size: float = Field(default=0.5, gt=0)
    goal_tolerance: float = Field(default=0.5, ge=0)
    goal_bias: float = Field(default=0.05, ge=0, le=1)
    seed: int | None = None

    @field_validator("sampler")
    @classmethod
    def validate_sampler(cls, v: type[Sampler]) -> type[Sampler]:
        if not isinstance(v, type):
            raise TypeError("sampler must be a class type")
        if not issubclass(v, Sampler):
            raise TypeError("sampler must inherit from Sampler")
        return v


class RRT(RRTBase):
    """RRT (Rapidly-exploring Random Tree) path planner."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRTConfig | None = None,
    ) -> None:
        """Initialize the RRT planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            config : RRTConfig
        """
        if config is None:
            config = RRTConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=config.max_iterations,
            step_size=config.step_size,
            goal_tolerance=config.goal_tolerance,
            seed=config.seed,
        )

        # Sampler
        sampler_kwargs = dict(config.sampler_kwargs)
        sampler_kwargs["bounds"] = bounds
        sampler_kwargs["seed"] = config.seed
        if config.sampler is GoalBiasedSampler:
            sampler_kwargs["goal_state"] = self.goal_state
            sampler_kwargs["goal_bias"] = config.goal_bias
        self.sampler = config.sampler(**sampler_kwargs)  # type: ignore[misc]

        # Tree
        self.nodes: list[Node] = []

    def plan(self) -> list[Node] | None:
        """Run the RRT algorithm to find a path.

        Returns:
            List of nodes from start to goal, or None if no path found
        """
        # Initialize tree with start node
        self.root = Node(state=self.start_state, parent=None, cost=0.0)
        self.graph.reset()
        self.graph.add_node(self.root)
        self.nodes = self.graph.nodes
        self.goal_node = None

        # Check if start and goal are collision-free
        if not self._check_start_goal_collision():
            return None
        root_goal = self._connect_goal(self.root)
        if root_goal is not None:
            self.goal_node = root_goal
            if root_goal is not self.root:
                self.graph.add_node(root_goal)
            return self._extract_path()

        # Main RRT loop
        for iteration in tqdm(range(self.max_iterations), desc="RRT Planning", unit="iter"):
            # Sample a random state
            random_state = self.sampler.sample()
            random_node = Node(state=random_state)

            # Find nearest node in the tree
            nearest_node = self.graph.nearest(random_node)

            # Steer towards the random state
            new_node, edge_cost = self.graph.steer(nearest_node, random_node, self.step_size)

            # Check if the path is collision-free
            if self.graph.is_edge_collision_free(nearest_node, new_node, self.collision_checker):
                # Add new node to the tree
                new_node.change_parent(nearest_node, nearest_node.cost + edge_cost)
                self.graph.add_node(new_node)

                # Check if goal is reached
                goal_candidate = self._connect_goal(new_node)
                if goal_candidate is not None:
                    self.goal_node = goal_candidate
                    if goal_candidate is not new_node:
                        self.nodes.append(goal_candidate)
                    print(f"Goal reached in {iteration + 1} iterations!")
                    return self._extract_path()

        print(f"Failed to reach goal after {self.max_iterations} iterations")
        return None

    def _extract_path(self) -> list[Node]:
        """Extract the path from start to goal.

        Returns:
            List of nodes from start to goal
        """
        if self.goal_node is None:
            return []

        return self.goal_node.get_path_from_root()

    def get_tree_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get all edges in the tree for visualization.

        Returns:
            List of (parent_state, child_state) tuples
        """
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append((node.parent.state, node.state))
        return edges

    def get_path_length(self) -> float:
        """Get the length of the path.

        Returns:
            Path length, or inf if no path found
        """
        if self.goal_node is None:
            return float("inf")
        return self.goal_node.cost

    def get_stats(self) -> dict:
        """Get statistics about the planning process.

        Returns:
            Dictionary with statistics
        """
        return {
            "num_nodes": len(self.nodes),
            "goal_reached": self.goal_node is not None,
            "path_length": self.get_path_length(),
            "path_nodes": len(self._extract_path()) if self.goal_node else 0,
        }

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the tree for visualization.

        Returns:
            List of all nodes explored during planning
        """
        return self.nodes

    def get_goal_node(self) -> Node | None:
        """Get the goal node for visualization.

        Returns:
            The node that reached the goal, or None if planning failed
        """
        return self.goal_node


class RRTConnectConfig(BaseModel):
    """Configuration for RRT-Connect algorithm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_iterations: int = Field(default=5000, ge=0)
    step_size: float = Field(default=0.5, gt=0)
    goal_tolerance: float = Field(default=0.5, ge=0)
    space: PlanningSpace | None = None
    seed: int | None = None


class RRTConnect(RRTBase):
    """RRT-Connect algorithm (bidirectional RRT)."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRTConnectConfig | None = None,
    ) -> None:
        """Initialize the RRT-Connect planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance
            config: RRTConnectConfig instance
        """
        if config is None:
            config = RRTConnectConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=config.max_iterations,
            step_size=config.step_size,
            goal_tolerance=config.goal_tolerance,
            seed=config.seed,
        )

        # Sampler
        self.sampler = UniformSampler(bounds=bounds, seed=config.seed)

        # Trees
        self.space = EuclideanSpace() if config.space is None else config.space
        self.graph.space = self.space
        self.start_graph = Graph(self.space)
        self.goal_graph = Graph(self.space)
        self.start_nodes: list[Node] = []
        self.goal_nodes: list[Node] = []
        self.start_root: Node | None = None
        self.goal_root: Node | None = None
        self.connection_point_start: Node | None = None
        self.connection_point_goal: Node | None = None
        self.swapped: bool = False  # Track if trees are swapped

        # Track failed extension attempts for visualization
        self.failed_nodes: list[Node] = []
        self.failed_edges: list[tuple[np.ndarray, np.ndarray]] = []

    def plan(self) -> list[Node] | None:
        """Run the RRT-Connect algorithm.

        Returns:
            List of nodes from start to goal, or None if no path found
        """
        # Initialize trees
        self.start_root = Node(state=self.start_state, parent=None, cost=0.0)
        self.goal_root = Node(state=self.goal_state, parent=None, cost=0.0)
        self.start_graph.reset()
        self.goal_graph.reset()
        self.start_graph.add_node(self.start_root)
        self.goal_graph.add_node(self.goal_root)
        self.start_nodes = self.start_graph.nodes
        self.goal_nodes = self.goal_graph.nodes
        self.failed_nodes = []  # Reset failed attempts
        self.failed_edges = []
        self.connection_point_start = None
        self.connection_point_goal = None
        self.swapped = False

        # Check if start and goal are collision-free
        if not self._check_start_goal_collision():
            return None
        start_goal_node = Node(state=self.goal_state)
        if self.graph.distance(
            self.start_root,
            start_goal_node,
        ) <= self.goal_tolerance and self.graph.is_edge_collision_free(
            self.start_root,
            start_goal_node,
            self.collision_checker,
        ):
            self.connection_point_start = self.start_root
            connection_goal = Node(state=self.start_state)
            connection_goal.change_parent(
                self.goal_root,
                self.goal_root.cost + self.goal_graph.edge_cost(self.goal_root, connection_goal),
            )
            self.goal_graph.add_node(connection_goal)
            self.connection_point_goal = connection_goal
            return self._extract_path()

        # Main loop
        for iteration in tqdm(range(self.max_iterations), desc="RRT-Connect Planning", unit="iter"):
            # Sample random state
            random_state = self.sampler.sample()
            random_node = Node(state=random_state)

            # Extend start tree
            new_node_start = self._extend_tree(self.start_graph, random_node)
            if new_node_start is not None:
                # Try to connect to goal tree
                connection_node = self._connect_tree(self.goal_graph, new_node_start)

                if connection_node is not None:
                    self.connection_point_start = new_node_start
                    self.connection_point_goal = connection_node
                    print(f"Trees connected in {iteration + 1} iterations!")
                    return self._extract_path()

            # Swap trees (alternate which tree extends)
            self.start_graph, self.goal_graph = self.goal_graph, self.start_graph
            self.start_nodes = self.start_graph.nodes
            self.goal_nodes = self.goal_graph.nodes
            self.swapped = not self.swapped

        print(f"Failed to connect trees after {self.max_iterations} iterations")
        return None

    def _extend_tree(self, tree: Graph, target: Node) -> Node | None:
        """Extend a tree towards a target.

        Args:
            tree: The tree to extend
            target: The target node

        Returns:
            The new node if successful, None otherwise
        """
        nearest = tree.nearest(target)
        new_node, edge_cost = tree.steer(nearest, target, self.step_size)

        if tree.is_edge_collision_free(nearest, new_node, self.collision_checker):
            new_node.change_parent(nearest, nearest.cost + edge_cost)
            tree.add_node(new_node)
            return new_node

        # Track failed extension attempt
        self._record_failed_extension(tree, new_node.state, nearest)
        return None

    def _connect_tree(self, tree: Graph, target: Node) -> Node | None:
        """Try to connect a tree to a target node.

        Args:
            tree: The tree to connect
            target: The target node (from the other tree)

        Returns:
            The last node added to this tree that connects to target, None otherwise
        """
        nearest = tree.nearest(target)

        while True:
            dist = tree.distance(nearest, target)

            if dist <= self.step_size:
                if tree.is_edge_collision_free(nearest, target, self.collision_checker):
                    target_node = Node(state=target.state)
                    target_node.change_parent(
                        nearest,
                        nearest.cost + tree.edge_cost(nearest, target_node),
                    )
                    tree.add_node(target_node)
                    return target_node
                self._record_failed_extension(tree, target.state, nearest)
                return None

            new_node, edge_cost = tree.steer(nearest, target, self.step_size)

            if not tree.is_edge_collision_free(nearest, new_node, self.collision_checker):
                self._record_failed_extension(tree, new_node.state, nearest)
                return None

            new_node.change_parent(nearest, nearest.cost + edge_cost)
            tree.add_node(new_node)
            prev = nearest
            nearest = new_node

            if np.allclose(prev.state, nearest.state):
                return None

    def _record_failed_extension(self, tree: Graph, state: np.ndarray, parent: Node) -> None:
        """Record a rejected extension without adding it to the tree."""
        failed_node = Node(state=state)
        failed_node.cost = parent.cost + tree.edge_cost(parent, failed_node)
        self.failed_nodes.append(failed_node)
        self.failed_edges.append((parent.state.copy(), failed_node.state.copy()))

    def _extract_path(self) -> list[Node]:
        """Extract the complete path from start to goal."""
        if self.connection_point_start is None or self.connection_point_goal is None:
            return []

        path_from_start = []
        current_node: Node | None = self.connection_point_start
        while current_node is not None:
            path_from_start.append(current_node)
            current_node = current_node.parent

        path_from_goal = []
        current_node = (
            self.connection_point_goal.parent
        )  # Skip connection point itself (same as connection_point_start)
        while current_node is not None:
            path_from_goal.append(current_node)
            current_node = current_node.parent

        if len(path_from_start) > 0 and np.allclose(path_from_start[-1].state, self.start_state):
            path_from_start.reverse()  # start → connection
            return path_from_start + path_from_goal
        else:
            path_from_goal.reverse()  # start → connection
            return path_from_goal + path_from_start

    def get_stats(self) -> dict:
        """Get statistics about the planning process.

        Returns:
            Dictionary with statistics
        """
        path = self._extract_path()
        return {
            "num_nodes_start": len(self.start_nodes),
            "num_nodes_goal": len(self.goal_nodes),
            "num_failed_attempts": len(self.failed_edges),
            "total_nodes": len(self.start_nodes) + len(self.goal_nodes) + len(self.failed_nodes),
            "trees_connected": self.connection_point_start is not None,
            "path_nodes": len(path),
        }

    def get_path_length(self) -> float:
        """Get the length of the path.

        Returns:
            Path length, or inf if no path found
        """
        path = self._extract_path()
        if not path:
            return float("inf")

        # Calculate total path length
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += self.graph.distance(path[i], path[i + 1])
        return total_length

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in both trees for visualization.

        Returns:
            List of all nodes explored during planning
        """
        return self.start_nodes + self.goal_nodes + self.failed_nodes

    def get_goal_node(self) -> Node | None:
        """Get the connection point for visualization.

        Returns:
            The connection point in the start tree, or None if trees didn't connect
        """
        # Return the connection point that's in the path to the goal
        if self.connection_point_start is None:
            return None

        # If trees were swapped, return the appropriate connection point
        if self.swapped:
            return self.connection_point_goal
        return self.connection_point_start


class RRTStarConfig(BaseModel):
    """Configuration for RRT* algorithm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sampler: type[Sampler] = GoalBiasedSampler
    sampler_kwargs: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = Field(default=5000, ge=0)
    step_size: float = Field(default=0.5, gt=0)
    goal_tolerance: float = Field(default=0.5, ge=0)
    goal_bias: float = Field(default=0.05, ge=0, le=1)
    radius_gain: float = Field(default=1.0, gt=0)
    return_first_solution: bool = True
    space: PlanningSpace | None = None
    seed: int | None = None

    @field_validator("sampler")
    @classmethod
    def validate_sampler(cls, v: type[Sampler]) -> type[Sampler]:
        if not isinstance(v, type):
            raise TypeError("sampler must be a class type")
        if not issubclass(v, Sampler):
            raise TypeError("sampler must inherit from Sampler")
        return v


class RRTStar(RRGBase):
    """RRT* (RRT-Star) path planner."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRTStarConfig | None = None,
    ) -> None:
        """Initialize the RRT* planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            config : RRTStarConfig
        """
        if config is None:
            config = RRTStarConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=config.max_iterations,
            step_size=config.step_size,
            goal_tolerance=config.goal_tolerance,
            radius_gain=config.radius_gain,
            seed=config.seed,
        )

        self.return_first_solution = config.return_first_solution
        self.space = EuclideanSpace() if config.space is None else config.space
        self.graph.space = self.space

        # Sampler
        sampler_kwargs = dict(config.sampler_kwargs)
        sampler_kwargs["bounds"] = bounds
        sampler_kwargs["seed"] = config.seed
        if config.sampler is GoalBiasedSampler:
            sampler_kwargs["goal_state"] = self.goal_state
            sampler_kwargs["goal_bias"] = config.goal_bias
        self.sampler = config.sampler(**sampler_kwargs)  # type: ignore[misc]

    def plan(self) -> list[Node] | None:
        """Run the RRT* algorithm.

        Returns:
            List of nodes from start to goal, or None if no path found
        """
        self.root = Node(state=self.start_state)
        self.path = None
        self.goal_node = None

        # graph initialization
        self.graph.reset()
        self.graph.add_node(self.root)

        if not self._check_start_goal_collision():
            return None
        root_goal = self._connect_goal(self.root)
        if root_goal is not None:
            self.goal_node = root_goal
            self.path = self._extract_path()
            return self.path

        goal_candidates: list[Node] = []

        # Main RRT* loop
        for iteration in tqdm(range(self.max_iterations), desc="RRT* Planning", unit="iter"):
            # Sample a random state
            random_state = self.sampler.sample()
            random_node = Node(state=random_state)

            # Find nearest node in the graph
            nearest_node = self.graph.nearest(random_node)
            new_node, _ = self.graph.steer(nearest_node, random_node, self.step_size)

            # Check if the path is collision-free
            if self.graph.is_edge_collision_free(
                nearest_node,
                new_node,
                self.collision_checker,
            ):
                neighbor_nodes = self.get_near_node(new_node)
                self.graph.add_node(new_node)

                min_cost_node = self.get_min_cost_node([*neighbor_nodes, nearest_node], new_node)

                if min_cost_node is None:
                    continue

                edge_cost = self.graph.edge_cost(min_cost_node, new_node)
                best_cost = min_cost_node.cost + edge_cost
                new_node.change_parent(min_cost_node, best_cost)
                self.graph.add_edge(min_cost_node, new_node, edge_cost)

                self._rewire_neighbors(new_node, neighbor_nodes)

                goal_candidate = self._connect_goal(new_node)
                if goal_candidate is not None:
                    if self.return_first_solution:
                        self.goal_node = goal_candidate
                        print(f"Goal reached in {iteration + 1} iterations!")
                        self.path = self._extract_path()
                        return self.path
                    else:
                        goal_candidates.append(goal_candidate)
                        if len(goal_candidates) == 1:
                            print(
                                f"First goal reached in {iteration + 1} iterations, continuing optimization..."
                            )

        if not self.return_first_solution and goal_candidates:
            self.goal_node = min(goal_candidates, key=lambda node: node.cost)
            print(
                f"Best goal found with cost {self.goal_node.cost:.3f} from {len(goal_candidates)} candidates"
            )
            self.path = self._extract_path()
            return self.path

        return None

    def _connect_goal(self, node: Node) -> Node | None:
        """Connect a near-goal node and register exact-goal edges in the graph."""
        goal_node = Node(state=self.goal_state)
        if self.graph.distance(node, goal_node) > self.goal_tolerance:
            return None
        if not self.graph.is_edge_collision_free(node, goal_node, self.collision_checker):
            return None
        if np.allclose(node.state, self.goal_state):
            return node

        edge_cost = self.graph.edge_cost(node, goal_node)
        goal_node.change_parent(node, node.cost + edge_cost)

        self.graph.add_node(goal_node)
        self.graph.add_edge(node, goal_node, edge_cost)
        return goal_node

    def _rewire_neighbors(self, new_node: Node, neighbor_nodes: list[Node]) -> None:
        """Rewire neighbor nodes if a better path through new_node exists.

        Args:
            new_node: The newly added node
            neighbor_nodes: List of nodes to potentially rewire
        """
        for neighbor_node in neighbor_nodes:
            if not self.graph.is_edge_collision_free(
                neighbor_node,
                new_node,
                self.collision_checker,
            ):
                continue

            edge_cost = self.graph.edge_cost(new_node, neighbor_node)

            # Rewiring
            if new_node.cost + edge_cost < neighbor_node.cost:
                old_parent = neighbor_node.parent
                if old_parent:
                    self.graph.remove_edge(neighbor_node, old_parent)
                self.graph.add_edge(new_node, neighbor_node, edge_cost)
                neighbor_node.change_parent(new_node, new_node.cost + edge_cost)

    def get_min_cost_node(self, nodes: list[Node], target: Node) -> Node | None:
        """Get the node from a list that provides the minimum cost to reach target."""
        min_cost = float("inf")
        min_cost_node = None

        for node in nodes:
            if not self.graph.is_edge_collision_free(node, target, self.collision_checker):
                continue

            cost = node.cost + self.graph.edge_cost(node, target)
            if cost < min_cost:
                min_cost = cost
                min_cost_node = node

        return min_cost_node

    def get_path_length(self) -> float:
        """Get the current path cost."""
        if self.goal_node is None:
            return float("inf")
        return self.goal_node.cost

    def get_stats(self) -> dict[str, float | int | bool | None]:
        """Get statistics about the planning process.

        Returns:
            Dictionary with number of nodes and edges
        """
        path_length = self.get_path_length()
        return {
            "num_nodes": len(self.graph.nodes),
            "goal_reached": self.goal_node is not None,
            "num_edges": len(self.graph.edges),
            "path_length": path_length if path_length > 0 else None,
            "path_nodes": len(self.path) if self.path else None,
        }

    def get_all_nodes(self) -> list[Node]:
        return self.graph.nodes

    def get_goal_node(self) -> Node | None:
        return self.goal_node

    def _extract_path(self) -> list[Node]:
        path = []
        node = self.goal_node
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class InformedRRTStar(RRTStar):
    """Informed RRT* path planner."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRTStarConfig | None = None,
    ) -> None:
        """Initialize the Informed RRT* planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            config: RRTStarConfig
        """
        if config is None:
            config = RRTStarConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            config=config,
        )
        self.informed_sampler = InformedSampler(
            bounds=bounds,
            start_state=self.start_state,
            goal_state=self.goal_state,
            seed=config.seed,
        )
        self.goal_nodes: list[Node] = []

    def plan(self) -> list[Node] | None:
        """Run the RRT* algorithm."""

        self.root = Node(state=self.start_state)
        self.path = None
        self.goal_node = None
        self.goal_nodes = []
        # graph initialization
        self.graph.reset()
        self.graph.add_node(self.root)
        c_best = float("inf")

        if not self._check_start_goal_collision():
            return None
        root_goal = self._connect_goal(self.root)
        if root_goal is not None:
            self.goal_node = root_goal
            self.goal_nodes.append(root_goal)
            self.path = self._extract_path()
            return self.path

        # Main RRT* loop
        for _ in tqdm(range(self.max_iterations), desc="Informed RRT* Planning", unit="iter"):
            # Sample a random state
            if self.goal_nodes:
                c_best = min(node.cost for node in self.goal_nodes)
                random_state = self.informed_sampler.sample(c_best)
            else:
                random_state = self.sampler.sample()

            random_node = Node(state=random_state)

            # Find nearest node in the graph
            nearest_node = self.graph.nearest(random_node)
            new_node, _ = self.graph.steer(nearest_node, random_node, self.step_size)

            # Check if the path is collision-free
            if self.graph.is_edge_collision_free(
                nearest_node,
                new_node,
                self.collision_checker,
            ):
                neighbor_nodes = self.get_near_node(new_node)
                self.graph.add_node(new_node)

                min_cost_node = self.get_min_cost_node([*neighbor_nodes, nearest_node], new_node)

                if min_cost_node is None:
                    continue

                edge_cost = self.graph.edge_cost(min_cost_node, new_node)
                best_cost = min_cost_node.cost + edge_cost
                new_node.change_parent(min_cost_node, best_cost)
                self.graph.add_edge(min_cost_node, new_node, edge_cost)

                # Rewire neighbors to use new_node if it provides a better path
                self._rewire_neighbors(new_node, neighbor_nodes)

                goal_candidate = self._connect_goal(new_node)
                if goal_candidate is not None:
                    if self.goal_node is None or goal_candidate.cost < self.goal_node.cost:
                        self.goal_node = goal_candidate
                    self.goal_nodes.append(goal_candidate)

        if self.goal_nodes:
            self.goal_node = min(self.goal_nodes, key=lambda node: node.cost)
            self.path = self._extract_path()
        else:
            self.path = None
        return self.path

    def get_stats(self) -> dict[str, float | int | bool | None]:
        """Get statistics about the planning process.

        Returns:
            Dictionary with number of nodes, edges, and Informed RRT* specific stats
        """
        base_stats = super().get_stats()
        base_stats["num_goal_candidates"] = len(self.goal_nodes)
        return base_stats
