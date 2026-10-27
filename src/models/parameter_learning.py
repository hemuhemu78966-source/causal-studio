"""
Parameter Estimation and Learning for Causal Bayesian Networks.

Fits Conditional Probability Distributions (CPDs) from observational data
using Maximum Likelihood Estimation (MLE) or Bayesian Estimation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from .bayesian_network import CausalBayesianNetwork


class ParameterLearning:
    """
    Fits and validates Conditional Probability Distributions (CPDs) for Causal Bayesian Networks.
    """

    def __init__(self, network: CausalBayesianNetwork):
        """
        Initialize with a CausalBayesianNetwork DAG structure.

        Args:
            network: CausalBayesianNetwork instance with nodes and edges defined.
        """
        self.network = network

    def fit_mle(self, data: pd.DataFrame) -> CausalBayesianNetwork:
        """
        Fit CPDs using Maximum Likelihood Estimation (MLE).

        Args:
            data: Tabular pandas DataFrame matching network nodes.

        Returns:
            The parameterized CausalBayesianNetwork.
        """
        model = BayesianNetwork(list(self.network.dag.edges()))
        for node in self.network.dag.nodes():
            if node not in model.nodes():
                model.add_node(node)

        estimator = MaximumLikelihoodEstimator(model, data)
        cpds = estimator.get_parameters()

        for cpd in cpds:
            self.network.set_cpd(cpd.variable, cpd)

        self.network.build_model()
        return self.network

    def fit_bayesian(self,
                     data: pd.DataFrame,
                     prior_type: str = "BDeu",
                     equivalent_sample_size: int = 10) -> CausalBayesianNetwork:
        """
        Fit CPDs using Bayesian Estimation with smoothing priors (avoids zero probabilities).

        Args:
            data: Tabular pandas DataFrame.
            prior_type: 'BDeu', 'dirichlet', or 'K2'
            equivalent_sample_size: Sample size weight for pseudo-counts.

        Returns:
            The parameterized CausalBayesianNetwork.
        """
        model = BayesianNetwork(list(self.network.dag.edges()))
        for node in self.network.dag.nodes():
            if node not in model.nodes():
                model.add_node(node)

        estimator = BayesianEstimator(model, data)
        cpds = estimator.get_parameters(
            prior_type=prior_type,
            equivalent_sample_size=equivalent_sample_size
        )

        for cpd in cpds:
            self.network.set_cpd(cpd.variable, cpd)

        self.network.build_model()
        return self.network

    def get_cpd_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a clean dictionary summary of all fitted CPDs in the network.

        Returns:
            Dictionary mapping variable names to their state names and probability matrices.
        """
        summary = {}
        for var, cpd in self.network.cpds.items():
            summary[var] = {
                "variable": cpd.variable,
                "variable_card": cpd.variable_card,
                "evidence": cpd.variables[1:],
                "evidence_card": cpd.cardinality[1:].tolist() if len(cpd.cardinality) > 1 else [],
                "values": cpd.values.tolist(),
                "state_names": getattr(cpd, 'state_names', None)
            }
        return summary
