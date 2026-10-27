"""
Causal Bayesian Network implementation.

This module provides the core structure for representing and manipulating
causal Bayesian networks.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


class CausalBayesianNetwork:
    """
    A Causal Bayesian Network for modeling causal relationships.

    Attributes:
        dag: Directed acyclic graph representing causal structure
        cpds: Conditional probability distributions for each node
    """

    def __init__(self):
        """Initialize an empty causal Bayesian network."""
        self.dag = nx.DiGraph()
        self.model = None
        self.cpds = {}

    def add_node(self, node: str, cardinality: int = 2) -> None:
        """
        Add a node to the network.

        Args:
            node: Node name
            cardinality: Number of possible values for this variable
        """
        self.dag.add_node(node, cardinality=cardinality)

    def add_nodes(self, nodes: List[str], cardinalities: Optional[Dict[str, int]] = None) -> None:
        """
        Add multiple nodes to the network.

        Args:
            nodes: List of node names
            cardinalities: Optional dictionary mapping node names to cardinalities
        """
        for node in nodes:
            card = cardinalities.get(node, 2) if cardinalities else 2
            self.add_node(node, card)

    def add_edge(self, parent: str, child: str) -> None:
        """
        Add a causal edge from parent to child.

        Args:
            parent: Parent node name
            child: Child node name

        Raises:
            ValueError: If edge would create a cycle
        """
        self.dag.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self.dag):
            self.dag.remove_edge(parent, child)
            raise ValueError(f"Adding edge {parent} -> {child} would create a cycle")

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        """
        Add multiple causal edges.

        Args:
            edges: List of (parent, child) tuples
        """
        for parent, child in edges:
            self.add_edge(parent, child)

    def set_cpd(self, node: str, cpd: TabularCPD) -> None:
        """
        Set the conditional probability distribution for a node.

        Args:
            node: Node name
            cpd: TabularCPD object from pgmpy
        """
        self.cpds[node] = cpd

    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes of a given node."""
        return list(self.dag.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        """Get child nodes of a given node."""
        return list(self.dag.successors(node))

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendant nodes of a given node."""
        return nx.descendants(self.dag, node)

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestor nodes of a given node."""
        return nx.ancestors(self.dag, node)

    def is_d_separated(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Check if X and Y are d-separated given Z.

        Args:
            X: First set of variables
            Y: Second set of variables
            Z: Conditioning set

        Returns:
            True if X and Y are d-separated given Z
        """
        separated = getattr(nx, 'is_d_separator', None) or nx.d_separated
        return separated(self.dag, set(X), set(Y), set(Z))

    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        Get the Markov blanket of a node.

        The Markov blanket includes parents, children, and parents of children.
        """
        parents = set(self.get_parents(node))
        children = set(self.get_children(node))
        spouses = set()
        for child in children:
            spouses.update(self.get_parents(child))
        spouses.discard(node)

        return parents | children | spouses

    def build_model(self) -> BayesianNetwork:
        """
        Build a pgmpy BayesianNetwork model from the current structure.

        Returns:
            BayesianNetwork object ready for inference
        """
        edges = list(self.dag.edges())
        self.model = BayesianNetwork(edges)
        for node in self.dag.nodes():
            if node not in self.model.nodes():
                self.model.add_node(node)

        if self.cpds:
            self.model.add_cpds(*self.cpds.values())
            self.model.check_model()

        return self.model

    def validate(self) -> bool:
        """
        Validate the network structure and CPDs.

        Returns:
            True if the network is valid
        """
        if not nx.is_directed_acyclic_graph(self.dag):
            return False

        if self.model:
            return self.model.check_model()

        return True

    def get_topological_order(self) -> List[str]:
        """Get nodes in topological order."""
        return list(nx.topological_sort(self.dag))

    def __repr__(self) -> str:
        """String representation of the network."""
        n_nodes = self.dag.number_of_nodes()
        n_edges = self.dag.number_of_edges()
        return f"CausalBayesianNetwork(nodes={n_nodes}, edges={n_edges})"
