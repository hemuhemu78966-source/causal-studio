"""
Unit tests for Causal Bayesian Network implementation.
"""

import pytest
import numpy as np
from src.models.bayesian_network import CausalBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


class TestCausalBayesianNetwork:
    """Test suite for CausalBayesianNetwork class."""

    def test_initialization(self):
        """Test network initialization."""
        cbn = CausalBayesianNetwork()
        assert cbn.dag.number_of_nodes() == 0
        assert cbn.dag.number_of_edges() == 0

    def test_add_node(self):
        """Test adding a single node."""
        cbn = CausalBayesianNetwork()
        cbn.add_node('A')
        assert 'A' in cbn.dag.nodes()
        assert cbn.dag.nodes['A']['cardinality'] == 2

    def test_add_nodes(self):
        """Test adding multiple nodes."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['X', 'Y', 'Z'])
        assert cbn.dag.number_of_nodes() == 3
        assert 'X' in cbn.dag.nodes()
        assert 'Y' in cbn.dag.nodes()
        assert 'Z' in cbn.dag.nodes()

    def test_add_edge(self):
        """Test adding edges."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['A', 'B'])
        cbn.add_edge('A', 'B')
        assert cbn.dag.has_edge('A', 'B')
        assert cbn.dag.number_of_edges() == 1

    def test_cycle_detection(self):
        """Test that cycles are prevented."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['A', 'B', 'C'])
        cbn.add_edge('A', 'B')
        cbn.add_edge('B', 'C')

        with pytest.raises(ValueError):
            cbn.add_edge('C', 'A')  # This would create a cycle

    def test_get_parents(self):
        """Test getting parent nodes."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['A', 'B', 'C'])
        cbn.add_edges([('A', 'C'), ('B', 'C')])

        parents = cbn.get_parents('C')
        assert set(parents) == {'A', 'B'}

    def test_get_children(self):
        """Test getting child nodes."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['A', 'B', 'C'])
        cbn.add_edges([('A', 'B'), ('A', 'C')])

        children = cbn.get_children('A')
        assert set(children) == {'B', 'C'}

    def test_topological_order(self):
        """Test topological ordering."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['A', 'B', 'C'])
        cbn.add_edges([('A', 'B'), ('B', 'C')])

        order = cbn.get_topological_order()
        assert order.index('A') < order.index('B')
        assert order.index('B') < order.index('C')

    def test_markov_blanket(self):
        """Test Markov blanket computation."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['A', 'B', 'C', 'D'])
        cbn.add_edges([('A', 'B'), ('C', 'B'), ('B', 'D')])

        mb = cbn.get_markov_blanket('B')
        assert 'A' in mb  # parent
        assert 'C' in mb  # parent
        assert 'D' in mb  # child

    def test_validation(self):
        """Test network validation."""
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['A', 'B'])
        cbn.add_edge('A', 'B')

        assert cbn.validate()


class TestStructuralCausalModel:
    """Test suite for StructuralCausalModel."""

    def test_initialization(self):
        """Test SCM initialization."""
        from src.models.structural_causal_model import StructuralCausalModel

        scm = StructuralCausalModel()
        assert scm.network is not None
        assert len(scm.structural_equations) == 0

    def test_add_equation(self):
        """Test adding structural equations."""
        from src.models.structural_causal_model import StructuralCausalModel

        scm = StructuralCausalModel()

        def eq_x(parents, noise):
            return noise

        scm.add_structural_equation('X', eq_x)
        assert 'X' in scm.structural_equations

    def test_sampling(self):
        """Test sampling from SCM."""
        from src.models.structural_causal_model import StructuralCausalModel

        cbn = CausalBayesianNetwork()
        cbn.add_nodes(['X', 'Y'])
        cbn.add_edge('X', 'Y')

        scm = StructuralCausalModel(cbn)

        scm.add_structural_equation(
            'X',
            lambda parents, noise: (noise > 0).astype(int)
        )

        scm.add_structural_equation(
            'Y',
            lambda parents, noise: (parents.get('X', np.array([0])) + noise > 0).astype(int)
        )

        samples = scm.sample(n_samples=10)
        assert 'X' in samples
        assert 'Y' in samples
        assert len(samples['X']) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
