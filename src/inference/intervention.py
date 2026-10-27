"""
Intervention analysis for causal Bayesian networks.

Implements do-calculus and intervention operations for causal inference.
"""

import numpy as np
import networkx as nx
import itertools
from typing import Dict, List, Optional, Any, Set
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from ..models.bayesian_network import CausalBayesianNetwork


class InterventionAnalysis:
    """
    Perform intervention analysis using do-calculus.

    Interventions represent actions that force variables to specific values,
    breaking incoming causal links (graph mutilation).
    """

    def __init__(self, network: CausalBayesianNetwork):
        """
        Initialize intervention analysis.

        Args:
            network: CausalBayesianNetwork to perform interventions on
        """
        self.network = network
        self.inference_engine = None

        if network.model:
            self.inference_engine = VariableElimination(network.model)

    def do_intervention(self,
                        interventions: Dict[str, Any],
                        target: str,
                        evidence: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Compute P(target | do(interventions), evidence).

        Args:
            interventions: Variables to intervene on and their values
            target: Target variable to query
            evidence: Optional observational evidence

        Returns:
            Probability distribution over target
        """
        if target not in self.network.cpds:
            raise ValueError(f"Unknown target: {target}")
        for node, value in {**(evidence or {}), **interventions}.items():
            if node not in self.network.cpds:
                raise ValueError(f"Unknown variable: {node}")
            if value not in self.network.cpds[node].state_names[node]:
                raise ValueError(f"Invalid state {value!r} for {node}")
        if target in interventions:
            return {str(s): float(s == interventions[target])
                    for s in self.network.cpds[target].state_names[target]}
        if not interventions:
            # Observational query
            if self.network.model is None:
                self.network.build_model()
            engine = VariableElimination(self.network.model)
            valid_evidence = {k: v for k, v in (evidence or {}).items() if k != target}
            res = engine.query(variables=[target], evidence=valid_evidence if valid_evidence else None)
            return self._format_result(res, target)

        # Create mutilated graph (remove incoming edges to intervention nodes and set deterministic CPDs)
        mutilated_network = self._create_mutilated_graph(interventions)
        mutilated_inference = VariableElimination(mutilated_network.model)

        valid_evidence = {}
        if evidence:
            for k, v in evidence.items():
                if k != target and k not in interventions:
                    valid_evidence[k] = v

        result = mutilated_inference.query(
            variables=[target],
            evidence=valid_evidence if valid_evidence else None
        )

        return self._format_result(result, target)

    def _format_result(self, result: Any, target: str) -> Dict[str, float]:
        """Format pgmpy DiscreteFactor result into a clean dictionary."""
        if hasattr(result, 'state_names') and target in result.state_names:
            states = result.state_names[target]
        else:
            states = list(range(len(result.values)))
        return {str(val): float(prob) for val, prob in zip(states, result.values)}

    def _create_mutilated_graph(self, interventions: Dict[str, Any]) -> CausalBayesianNetwork:
        """
        Create a mutilated graph by removing incoming edges to intervened variables
        and assigning deterministic point-mass CPDs P(X=x) = 1.0.

        Args:
            interventions: Variables being intervened on

        Returns:
            New CausalBayesianNetwork with graph mutilation applied
        """
        mutilated = CausalBayesianNetwork()

        # 1. Copy all nodes
        for node in self.network.dag.nodes():
            card = self.network.dag.nodes[node].get('cardinality', 2)
            mutilated.add_node(node, card)

        # 2. Copy edges EXCEPT those pointing INTO intervened variables (breaking parent links)
        for parent, child in self.network.dag.edges():
            if child not in interventions:
                mutilated.add_edge(parent, child)

        # 3. For non-intervened nodes, copy their original CPDs (their parents are unchanged)
        for node, cpd in self.network.cpds.items():
            if node not in interventions:
                mutilated.set_cpd(node, cpd)

        # 4. For intervened nodes, assign a deterministic prior CPD: P(node = val) = 1.0
        for node, val in interventions.items():
            card = self.network.dag.nodes[node].get('cardinality', 2)
            if node in self.network.cpds:
                card = self.network.cpds[node].variable_card

            probs = np.zeros((card, 1))
            states = list(self.network.cpds[node].state_names[node])
            if val not in states:
                raise ValueError(f"Invalid state {val!r} for {node}")
            probs[states.index(val), 0] = 1.0
            det_cpd = TabularCPD(variable=node, variable_card=card, values=probs,
                                 state_names={node: states})
            mutilated.set_cpd(node, det_cpd)

        mutilated.build_model()
        return mutilated

    def average_treatment_effect(self,
                                 treatment: str,
                                 outcome: str,
                                 confounders: Optional[List[str]] = None) -> float:
        """
        Compute Average Treatment Effect (ATE).

        ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
        """
        p_y1 = self.do_intervention({treatment: 1}, outcome)
        expected_y1 = sum(float(k) * v for k, v in p_y1.items())

        p_y0 = self.do_intervention({treatment: 0}, outcome)
        expected_y0 = sum(float(k) * v for k, v in p_y0.items())

        return expected_y1 - expected_y0

    def conditional_treatment_effect(self,
                                     treatment: str,
                                     outcome: str,
                                     condition: Dict[str, Any]) -> float:
        """
        Compute Conditional Average Treatment Effect (CATE).

        CATE = E[Y | do(T=1), X=x] - E[Y | do(T=0), X=x]
        """
        p_y1 = self.do_intervention({treatment: 1}, outcome, evidence=condition)
        expected_y1 = sum(float(k) * v for k, v in p_y1.items())

        p_y0 = self.do_intervention({treatment: 0}, outcome, evidence=condition)
        expected_y0 = sum(float(k) * v for k, v in p_y0.items())

        return expected_y1 - expected_y0

    def find_adjustment_set(self,
                            treatment: str,
                            outcome: str) -> Optional[Set[str]]:
        """
        Find a minimal adjustment set for identifying causal effects.
        """
        graph = self.network.dag.copy()
        graph.remove_edges_from(list(graph.out_edges(treatment)))
        candidates = sorted(set(graph.nodes()) - {treatment, outcome}
                            - nx.descendants(self.network.dag, treatment))
        separated = getattr(nx, 'is_d_separator', None) or nx.d_separated
        for size in range(len(candidates) + 1):
            for subset in itertools.combinations(candidates, size):
                if separated(graph, {treatment}, {outcome}, set(subset)):
                    return set(subset)
        return None

    def _find_backdoor_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all backdoor paths from source to target."""
        backdoor_paths = []
        undirected = self.network.dag.to_undirected()

        for path in nx.all_simple_paths(undirected, source, target):
            if len(path) > 1 and self.network.dag.has_edge(path[1], path[0]):
                backdoor_paths.append(path)

        return backdoor_paths

    def _blocks_all_paths(self, adjustment_set: Set[str], paths: List[List[str]]) -> bool:
        """Check if adjustment set blocks all given paths."""
        for path in paths:
            if not any(node in adjustment_set for node in path[1:-1]):
                return False
        return True
