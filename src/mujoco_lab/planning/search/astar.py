"""A* (Dijkstra-style) path planning algorithm using only edge cost."""

import heapq

import numpy as np

from ..graph import Graph, Node


class AStar:
    """A* (Dijkstra-style) path planning algorithm using only edge cost."""

    def __init__(self, graph: Graph) -> None:
        """Initialize A* planner.

        Args:
            graph: Graph instance containing nodes and edges
        """
        self.graph = graph

    def get_neighbors(self, node: Node) -> list[tuple[Node, float]]:
        """Return neighbors of a node and edge costs."""
        neighbors = []
        for edge in self.graph.edges:
            if edge.contains_node(node):
                other_node = edge.get_other_node(node)
                neighbors.append((other_node, edge.cost))
        return neighbors

    def search(self, start: Node, goal: Node) -> list[Node]:
        """Find the lowest-cost path from start to goal using edge cost only."""
        open_set: list[tuple[float, Node]] = []
        heapq.heappush(open_set, (0.0, start))

        came_from: dict[Node, Node | None] = {start: None}
        g_score = {node: float("inf") for node in self.graph.nodes}
        g_score[start] = 0.0

        while open_set:
            _, current = heapq.heappop(open_set)  # _ : current cost

            if np.allclose(current.state, goal.state, atol=1e-6):
                return self._reconstruct_path(came_from, current)

            for neighbor, edge_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + edge_cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g, neighbor))

        # No path found
        return []

    def _reconstruct_path(self, came_from: dict[Node, Node | None], current: Node) -> list[Node]:
        """Reconstruct path from start to goal."""
        path = [current]
        parent = came_from[current]
        while parent is not None:
            current = parent
            path.append(current)
            parent = came_from[current]
        path.reverse()
        return path
