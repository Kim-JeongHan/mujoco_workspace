"""Graph class for managing planning environment."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from ..collision import CollisionChecker
from ..space import EuclideanSpace, PlanningSpace
from .node import Node


class Edge:
    """Undirected edge class for managing planning environment."""

    def __init__(self, node1: Node, node2: Node, cost: float = 0.0) -> None:
        """Initialize the undirected edge.

        Args:
            node1: First node
            node2: Second node
            cost: Cost of the edge
        """
        self.node1 = node1
        self.node2 = node2
        self.cost = cost

    def __eq__(self, other: object) -> bool:
        """Check equality of two edges (undirected).

        Two edges are equal if they connect the same nodes regardless of order.
        """
        if not isinstance(other, Edge):
            return False
        return (self.node1 is other.node1 and self.node2 is other.node2) or (
            self.node1 is other.node2 and self.node2 is other.node1
        )

    def __hash__(self) -> int:
        """Return hash of the edge (undirected).

        Hash is the same regardless of node order.
        """
        # Use frozenset to ensure order-independent hashing
        return hash(frozenset([id(self.node1), id(self.node2)]))

    def contains_node(self, node: Node) -> bool:
        """Check if the edge contains a given node.

        Args:
            node: Node to check

        Returns:
            True if the edge contains the node
        """
        return self.node1 is node or self.node2 is node

    def get_other_node(self, node: Node) -> Node:
        """Get the other node in the edge.

        Args:
            node: One node in the edge

        Returns:
            The other node

        Raises:
            ValueError: If the given node is not in the edge
        """
        if self.node1 is node:
            return self.node2
        elif self.node2 is node:
            return self.node1
        else:
            raise ValueError("Node is not in the edge")

    def __repr__(self) -> str:
        """Return string representation of the edge."""
        return f"Edge(node1={self.node1}, node2={self.node2}, cost={self.cost})"


class Graph:
    """Graph class for managing planning environment."""

    def __init__(self, space: PlanningSpace | None = None) -> None:
        """Initialize the graph."""
        self.space = EuclideanSpace() if space is None else space
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: The node to add
        """
        self.nodes.append(node)

    def add_edge(self, node1: Node, node2: Node, cost: float) -> None:
        """Add an edge to the graph.

        Args:
            node1: The first node
            node2: The second node
            cost: The cost of the edge
        """
        new_edge = Edge(node1, node2, cost)
        # Check if edge already exists (order-independent)
        if new_edge not in self.edges:
            self.edges.append(new_edge)

    def remove_edge(self, node1: Node, node2: Node) -> None:
        """Remove an edge from the graph (undirected).

        Args:
            node1: The first node
            node2: The second node
        """
        edge_to_remove = Edge(node1, node2)
        if edge_to_remove in self.edges:
            self.edges.remove(edge_to_remove)

    def reset(self) -> None:
        """Reset the graph."""
        self.nodes = []
        self.edges = []

    def get_nodes_num(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self.nodes)

    def get_edges_num(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.edges)

    def distance(self, node1: Node, node2: Node) -> float:
        """Return the planning-space distance between nodes."""
        return self.space.distance(node1.state, node2.state)

    def nearest(self, target: Node) -> Node:
        """Find the nearest graph node according to the graph planning space."""
        if not self.nodes:
            raise ValueError("Node list is empty")
        return min(self.nodes, key=lambda node: self.distance(node, target))

    def near(self, target: Node, radius: float) -> list[Node]:
        """Return graph nodes within radius according to the graph planning space."""
        return [node for node in self.nodes if self.distance(node, target) <= radius]

    def steer(self, from_node: Node, to_node: Node, max_distance: float) -> tuple[Node, float]:
        """Steer from one node toward another using the graph planning space."""
        new_state = self.space.steer(from_node.state, to_node.state, max_distance)
        if new_state.shape != from_node.state.shape:
            raise ValueError("planning space steer must return a state with the node dimension")
        new_node = Node(state=new_state)
        new_cost = self.edge_cost(from_node, new_node)
        return new_node, new_cost

    def edge_cost(self, node1: Node, node2: Node) -> float:
        """Return the planning-space edge cost between nodes."""
        return self.space.edge_cost(node1.state, node2.state)

    def edge_states(self, node1: Node, node2: Node) -> np.ndarray:
        """Return planning-space edge states between nodes."""
        return self.space.edge_states(node1.state, node2.state)

    def is_edge_collision_free(
        self,
        node1: Node,
        node2: Node,
        collision_checker: CollisionChecker,
    ) -> bool:
        """Check every segment in the planning-space edge for collision."""
        states = self.edge_states(node1, node2)
        if states.ndim != 2 or states.shape[1] != node1.dim:
            raise ValueError("planning space edge states must have shape (N, node_dim)")
        if len(states) == 0:
            raise ValueError("planning space edge states must not be empty")
        if not np.allclose(states[0], node1.state):
            raise ValueError("planning space edge states must start at node1")
        if not np.allclose(states[-1], node2.state):
            raise ValueError("planning space edge states must end at node2")
        if len(states) == 1:
            return collision_checker.is_collision_free(states[0])
        return all(
            collision_checker.is_path_collision_free(from_state, to_state)
            for from_state, to_state in pairwise(states)
        )

    def check_edge(self, node1: Node, node2: Node) -> bool:
        """Check if an edge exists between two nodes."""
        return Edge(node1, node2) in self.edges

    def get_edge_by_node(self, node: Node) -> list[Edge]:
        """Get the edges of a node."""
        return [edge for edge in self.edges if edge.contains_node(node)]
