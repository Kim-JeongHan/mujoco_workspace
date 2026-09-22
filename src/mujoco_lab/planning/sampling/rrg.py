"""RRG (Rapidly-exploring Random Graph) algorithm implementation."""

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm

from ..collision import CollisionChecker
from ..graph import Node
from ..space import PlanningSpace
from .base import RRGBase
from .sampler import GoalBiasedSampler, Sampler


class RRGConfig(BaseModel):
    """Configuration for RRG algorithm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sampler: type[Sampler] = GoalBiasedSampler
    max_iterations: int = 1000
    radius_gain: float = 0.5  # gamma_RRG (scaling factor for connection radius)

    step_size: float = 0.5
    goal_tolerance: float = 0.5
    goal_bias: float = 0.05
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


class RRG(RRGBase):
    """RRG (Rapidly-exploring Random Graph) algorithm."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRGConfig | None = None,
    ) -> None:
        """Initialize the RRG planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance
            config: RRGConfig instance
        """
        if config is None:
            config = RRGConfig()

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

    def plan(self) -> list[Node] | None:
        """Run the RRG algorithm."""

        self.root = Node(state=self.start_state)
        self.path = None
        self.goal_node = None

        # graph initialization
        self.graph.reset()
        self.graph.add_node(self.root)

        if not self._check_start_goal_collision():
            return None

        # Main RRG loop
        for iteration in tqdm(range(self.max_iterations), desc="RRG Planning", unit="iter"):
            # Sample a random state
            random_state = self.sampler.sample()
            random_node = Node(state=random_state)

            # Find nearest node in the graph
            nearest_node = self.graph.nearest(random_node)
            new_node, new_cost = self.graph.steer(nearest_node, random_node, self.step_size)

            # Check if the path is collision-free
            if self.graph.is_edge_collision_free(nearest_node, new_node, self.collision_checker):
                self.graph.add_node(new_node)
                self.graph.add_edge(nearest_node, new_node, new_cost)

                neighbor_nodes = self.get_near_node(new_node)

                for neighbor_node in neighbor_nodes:
                    if self.graph.is_edge_collision_free(
                        neighbor_node,
                        new_node,
                        self.collision_checker,
                    ):
                        cost = self.graph.edge_cost(neighbor_node, new_node)
                        self.graph.add_edge(neighbor_node, new_node, cost)

                # Check if goal is reached (outside neighbor loop)
                if self._is_goal_reached(new_node):
                    self.goal_node = new_node
                    print(f"Goal reached in {iteration + 1} iterations!")
                    self.path = self.astar.search(self.root, self.goal_node)
                    return self.path

        return None

    def _is_goal_reached(self, node: Node) -> bool:
        """Check if a node is close enough to the goal.

        Args:
            node: The node to check

        Returns:
            True if within goal tolerance
        """
        return self.graph.distance(node, Node(state=self.goal_state)) <= self.goal_tolerance

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
