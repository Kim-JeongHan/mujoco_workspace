"""PRM (Probabilistic Roadmap Method) implementation."""

import numpy as np
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from ..collision import CollisionChecker
from ..graph import Node
from ..space import PlanningSpace
from .base import RRGBase
from .sampler import GoalBiasedSampler, Sampler, UniformSampler


class PRMConfig(BaseModel):
    """PRM configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sampler: type[Sampler] = UniformSampler
    sample_number: int = 1000
    max_retries: int = 5
    step_size: float = 0.1
    goal_tolerance: float = 0.1
    goal_bias: float = 0.05
    radius: float = 1.0
    space: PlanningSpace | None = None
    seed: int = 42


class PRM(RRGBase):
    """PRM (Probabilistic Roadmap Method) path planner."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: PRMConfig | None = None,
    ) -> None:
        """Initialize the PRM planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            config : PRMConfig
        """
        if config is None:
            config = PRMConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            step_size=config.step_size,
            goal_tolerance=config.goal_tolerance,
            seed=config.seed,
        )
        if config.space is not None:
            self.graph.space = config.space

        # Sampler
        if config.sampler is GoalBiasedSampler:
            self.sampler = config.sampler(  # type: ignore[call-arg]
                bounds=bounds,
                goal_state=self.goal_state,
                goal_bias=config.goal_bias,
                seed=config.seed,
            )
        else:
            self.sampler = config.sampler(bounds=bounds, seed=config.seed)

        # PRM parameters
        self.radius = config.radius
        self.sample_number = config.sample_number
        self.max_retries = config.max_retries

    def plan(self) -> list[Node] | None:
        """Plan a path using the PRM algorithm."""

        self.root = Node(state=self.start_state)
        self.path = None
        self.goal_node = Node(state=self.goal_state)

        # graph initialization
        self.graph.reset()
        self.graph.add_node(self.root)
        self.graph.add_node(self.goal_node)

        if not self._check_start_goal_collision():
            return None

        # Preprocess the graph
        for retry in tqdm(range(self.max_retries), desc="PRM Preprocessing", unit="retry"):
            for _ in tqdm(
                range(self.sample_number),
                desc=f"  Retry {retry + 1}/{self.max_retries}",
                unit="sample",
                leave=False,
            ):
                # Sample a random state
                random_state = self.sampler.sample()
                random_node = Node(state=random_state)

                if self.collision_checker.is_collision_free(random_node.state):
                    self.graph.add_node(random_node)

                    neighbor_nodes = self.get_near_node(random_node)
                    if not neighbor_nodes:  # no neighbor nodes
                        continue

                    sort_neighbor_nodes = sorted(
                        neighbor_nodes, key=lambda node: self.graph.distance(node, random_node)
                    )

                    for neighbor_node in sort_neighbor_nodes:
                        if self.graph.check_edge(neighbor_node, random_node):
                            continue

                        if self.graph.is_edge_collision_free(
                            neighbor_node,
                            random_node,
                            self.collision_checker,
                        ):
                            self.graph.add_edge(
                                neighbor_node,
                                random_node,
                                self.graph.edge_cost(neighbor_node, random_node),
                            )
            # Check connectivity from root to goal; only stop when a path exists
            candidate_path = self.astar.search(self.root, self.goal_node)
            if candidate_path:
                self.path = candidate_path
                break
            else:
                print(f"No path found after {self.sample_number} samples; retrying...")
        else:
            print(f"❌ Failed to connect to goal after {self.max_retries} retries.")
            return None

        return self.path

    def _connection_radius(self) -> float:
        """Return PRM's fixed connection radius."""
        return self.radius

    def get_stats(self) -> dict[str, float | int | bool | None]:
        """Get statistics about the planning process.

        Returns:
            Dictionary with number of nodes, edges, and path information
        """
        path_length = self.get_path_length()
        return {
            "num_nodes": len(self.graph.nodes),
            "goal_reached": self.path is not None,
            "num_edges": len(self.graph.edges),
            "path_length": path_length if path_length < float("inf") else None,
            "path_nodes": len(self.path) if self.path else None,
        }

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the roadmap.

        Returns:
            List of all nodes in the graph
        """
        return self.graph.nodes

    def get_goal_node(self) -> Node | None:
        """Get the goal node.

        Returns:
            The goal node if path was found, None otherwise
        """
        return self.goal_node if self.path is not None else None


class PRMStarConfig(PRMConfig):
    """PRM* configuration."""

    radius_gain: float = 5.0


class PRMStar(PRM):
    """PRM* (Probabilistic Roadmap Method Star) path planner."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: PRMStarConfig | None = None,
    ) -> None:
        """Initialize the PRM* planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            config: PRMStarConfig instance for algorithm parameters
        """
        if config is None:
            config = PRMStarConfig()
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            config=config,
        )

        self.radius_gain = config.radius_gain

    def _connection_radius(self) -> float:
        """Return PRM*'s asymptotically optimal dynamic connection radius."""
        return RRGBase._connection_radius(self)
