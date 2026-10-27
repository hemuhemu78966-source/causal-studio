"""
Automated Causal Discovery and Structure Learning.

Implements constraint-based (PC algorithm) and score-based (Hill-Climb with BIC/BDeu)
algorithms to discover causal DAG structures directly from observational data.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple, Dict, Any, Union

from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch, TreeSearch
try:
    from pgmpy.estimators import BIC as BicScore, BDeu as BDeuScore, K2 as K2Score
except ImportError:
    try:
        from pgmpy.estimators import BicScore, BDeuScore, K2Score
    except ImportError:
        from pgmpy.structure_score import BIC as BicScore, BDeu as BDeuScore, K2 as K2Score
from pgmpy.base import DAG
from ..models.bayesian_network import CausalBayesianNetwork


class CausalDiscovery:
    """
    Automated Structure Learning and Causal Discovery engine.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize Causal Discovery engine with tabular data.

        Args:
            data: pandas DataFrame containing discrete / discretized variables.
        """
        self.data = data.copy()
        self.learned_dag: Optional[CausalBayesianNetwork] = None
        self.raw_model: Optional[DAG] = None

    def discover_pc(self,
                    ci_test: str = "chi_square",
                    significance_level: float = 0.05,
                    variant: str = "stable",
                    max_cond_vars: int = 4) -> CausalBayesianNetwork:
        """
        Discover causal structure using the constraint-based PC algorithm.

        Args:
            ci_test: Conditional independence test ('chi_square', 'g_sq', 'log_likelihood')
            significance_level: Significance threshold for independence tests
            variant: PC algorithm variant ('orig', 'stable', 'parallel')
            max_cond_vars: Maximum conditioning set size to test

        Returns:
            CausalBayesianNetwork with discovered edges
        """
        est = PC(data=self.data)
        try:
            model = est.estimate(
                variant=variant,
                ci_test=ci_test,
                significance_level=significance_level,
                max_cond_vars=max_cond_vars,
                return_type="dag",
                show_progress=False
            )
        except Exception:
            try:
                model = est.estimate(
                    variant="stable",
                    ci_test="chi_square",
                    significance_level=significance_level,
                    show_progress=False
                )
            except Exception:
                # Fallback to building skeleton and orienting manually
                skel, sep = est.build_skeleton(ci_test="chi_square", significance_level=significance_level, show_progress=False)
                model = est.orient_colliders(skel, sep)

        dag = self._ensure_dag(model)
        self.raw_model = dag
        self.learned_dag = self._to_cbn(dag)
        return self.learned_dag

    def discover_hill_climb(self,
                            scoring_method: str = "bic",
                            black_list: Optional[List[Tuple[str, str]]] = None,
                            white_list: Optional[List[Tuple[str, str]]] = None,
                            max_iter: int = 50,
                            epsilon: float = 1e-4,
                            equivalent_sample_size: int = 10) -> CausalBayesianNetwork:
        """
        Discover causal structure using Score-Based Hill Climbing search.

        Args:
            scoring_method: 'bic', 'bdeu', or 'k2'
            black_list: List of forbidden directed edges (parent, child)
            white_list: List of required directed edges (parent, child)
            max_iter: Maximum search iterations
            epsilon: Score improvement threshold
            equivalent_sample_size: Hyperparameter for BDeu score

        Returns:
            CausalBayesianNetwork with discovered edges
        """
        if scoring_method.lower() == "bic":
            scorer = BicScore(data=self.data)
        elif scoring_method.lower() == "bdeu":
            scorer = BDeuScore(data=self.data, equivalent_sample_size=equivalent_sample_size)
        elif scoring_method.lower() == "k2":
            scorer = K2Score(data=self.data)
        else:
            raise ValueError(f"Unsupported scoring method: {scoring_method}")

        hc = HillClimbSearch(data=self.data)
        try:
            best_model = hc.estimate(
                scoring_method=scorer,
                max_iter=max_iter,
                epsilon=epsilon, max_indegree=3, show_progress=False
            )
        except TypeError:
            best_model = hc.estimate(
                scoring_method=scorer,
                max_iter=max_iter, max_indegree=3, show_progress=False
            )

        self.raw_model = best_model
        self.learned_dag = self._to_cbn(best_model)
        return self.learned_dag

    def discover_tree(self,
                      root_node: Optional[str] = None,
                      estimator_type: str = "tan",
                      class_node: Optional[str] = None) -> CausalBayesianNetwork:
        """
        Discover a Tree-augmented or Chow-Liu causal structure.

        Args:
            root_node: Optional root node for Chow-Liu tree
            estimator_type: 'chow-liu' or 'tan' (Tree-Augmented Naive Bayes)
            class_node: Class node for TAN estimator

        Returns:
            CausalBayesianNetwork
        """
        if estimator_type.lower() == 'tan' and root_node is None:
            root_node = next(c for c in self.data.columns if c != class_node)
        ts = TreeSearch(data=self.data, root_node=root_node, n_jobs=1)
        if estimator_type.lower() == "tan":
            if not class_node:
                raise ValueError("class_node must be specified for TAN estimator")
            model = ts.estimate(estimator_type="tan", class_node=class_node, show_progress=False)
        else:
            model = ts.estimate(estimator_type="chow-liu", show_progress=False)

        self.raw_model = model
        self.learned_dag = self._to_cbn(model)
        return self.learned_dag

    def evaluate_structure(self, true_dag: Optional[Union[nx.DiGraph, CausalBayesianNetwork]] = None) -> Dict[str, Any]:
        """
        Evaluate the learned structure using BIC score and optionally Structural Hamming Distance (SHD).

        Args:
            true_dag: Optional ground-truth causal DAG to calculate SHD, precision, recall

        Returns:
            Dictionary of structural quality metrics
        """
        if self.learned_dag is None:
            raise ValueError("No structure has been learned yet. Run discover_* first.")

        bic_scorer = BicScore(data=self.data)
        current_edges = list(self.learned_dag.dag.edges())
        pgmpy_dag = DAG()
        pgmpy_dag.add_nodes_from(self.data.columns)
        pgmpy_dag.add_edges_from(current_edges)
        score_val = bic_scorer.score(pgmpy_dag)

        metrics = {
            "bic_score": float(score_val),
            "num_nodes": self.learned_dag.dag.number_of_nodes(),
            "num_edges": self.learned_dag.dag.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.learned_dag.dag)
        }

        if true_dag is not None:
            if isinstance(true_dag, CausalBayesianNetwork):
                true_nx = true_dag.dag
            else:
                true_nx = true_dag

            true_edges = set(true_nx.edges())
            pred_edges = set(self.learned_dag.dag.edges())

            tp = len(true_edges & pred_edges)
            fp = len(pred_edges - true_edges)
            fn = len(true_edges - pred_edges)

            # Reverse edges count as 1 SHD error
            reversals = sum((v, u) in pred_edges for u, v in true_edges - pred_edges)
            shd = fp + fn - reversals

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics.update({
                "structural_hamming_distance": int(shd),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            })

        return metrics

    def _ensure_dag(self, graph: Any) -> DAG:
        """Helper to orient any undirected/partially directed edges safely to avoid cycles."""
        dag = DAG()
        dag.add_nodes_from(self.data.columns)
        
        edges = list(graph.edges())
        for u, v in edges:
            # Check if reverse already exists or adding u->v creates cycle
            if not dag.has_edge(v, u):
                dag.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(dag):
                    dag.remove_edge(u, v)
        return dag

    def _to_cbn(self, pgmpy_model: Any) -> CausalBayesianNetwork:
        """Convert a pgmpy DAG/BayesianNetwork to our CausalBayesianNetwork instance."""
        cbn = CausalBayesianNetwork()
        cardinalities = {col: int(self.data[col].nunique()) for col in self.data.columns}
        cbn.add_nodes(list(self.data.columns), cardinalities=cardinalities)

        for parent, child in pgmpy_model.edges():
            try:
                cbn.add_edge(parent, child)
            except ValueError:
                # Skip if cycle
                pass
        return cbn
