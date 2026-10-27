"""
Structural Causal Model (SCM) implementation.

SCMs extend Bayesian networks by explicitly modeling the functional
relationships between variables and exogenous noise terms.
"""

import numpy as np
from typing import Dict, Callable, List, Optional, Any
from .bayesian_network import CausalBayesianNetwork


class StructuralCausalModel:
    """
    Structural Causal Model for causal inference.

    An SCM consists of:
    - Endogenous variables (modeled causally)
    - Exogenous variables (external noise/randomness)
    - Structural equations defining relationships
    """

    def __init__(self, causal_network: Optional[CausalBayesianNetwork] = None):
        """
        Initialize an SCM.

        Args:
            causal_network: Optional CausalBayesianNetwork defining the structure
        """
        self.network = causal_network or CausalBayesianNetwork()
        self.structural_equations: Dict[str, Callable] = {}
        self.exogenous_distributions: Dict[str, Callable] = {}

    def add_structural_equation(self,
                               variable: str,
                               equation: Callable,
                               exogenous_dist: Optional[Callable] = None) -> None:
        """
        Add a structural equation for a variable.

        Args:
            variable: Variable name
            equation: Function that computes the variable value from parents and noise
            exogenous_dist: Distribution for exogenous noise (default: standard normal)
        """
        self.structural_equations[variable] = equation
        if exogenous_dist is None:
            exogenous_dist = lambda size=1: np.random.randn(size)
        self.exogenous_distributions[variable] = exogenous_dist

    def sample(self,
               n_samples: int = 1,
               interventions: Optional[Dict[str, Any]] = None,
               exogenous_values: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Sample from the SCM.

        Args:
            n_samples: Number of samples to generate
            interventions: Optional dict of variables to intervene on and their values
            exogenous_values: Optional dict of exogenous noise values (for counterfactuals)

        Returns:
            Dictionary mapping variable names to sampled values
        """
        interventions = interventions or {}

        # Generate or use provided exogenous values
        if exogenous_values is None:
            exogenous_values = {
                var: dist(size=n_samples)
                for var, dist in self.exogenous_distributions.items()
            }

        samples = {}

        # Process variables in topological order
        for var in self.network.get_topological_order():
            if var in interventions:
                # Hard intervention: set to intervention value
                value = interventions[var]
                if not isinstance(value, np.ndarray):
                    value = np.full(n_samples, value)
                samples[var] = value
            elif var in self.structural_equations:
                # Compute from structural equation
                parents = self.network.get_parents(var)
                parent_values = {p: samples[p] for p in parents}
                noise = exogenous_values.get(var, np.zeros(n_samples))
                samples[var] = self.structural_equations[var](parent_values, noise)
            else:
                # No equation defined, use only exogenous noise
                samples[var] = exogenous_values.get(var, np.zeros(n_samples))

        return samples

    def counterfactual(self,
                      observed: Dict[str, Any],
                      intervention: Dict[str, Any],
                      target: str) -> np.ndarray:
        """
        Compute counterfactual: "What would target be if we intervened,
        given what we observed?"

        This implements Pearl's three-step process:
        1. Abduction: Infer exogenous variables from observed data
        2. Action: Apply intervention
        3. Prediction: Compute target under intervention

        Args:
            observed: Observed variable values
            intervention: Intervention to apply
            target: Target variable to predict

        Returns:
            Counterfactual value of target
        """
        # Step 1: Abduction - infer exogenous variables
        exogenous = self._abduct(observed)

        # Steps 2 & 3: Action and Prediction - sample with intervention
        result = self.sample(
            n_samples=1,
            interventions=intervention,
            exogenous_values={k: np.array([v]) for k, v in exogenous.items()}
        )

        return result[target][0]

    def _abduct(self, observed: Dict[str, Any]) -> Dict[str, float]:
        """
        Abduction step: infer exogenous variables from observations.

        This is a simplified implementation. In practice, this may require
        solving a system of equations or using approximate inference.

        Args:
            observed: Observed variable values

        Returns:
            Inferred exogenous variable values
        """
        exogenous = {}

        # Process in topological order
        for var in self.network.get_topological_order():
            if var not in observed or var not in self.structural_equations:
                continue

            # Solve for exogenous noise: U = f_inv(X, parents(X))
            parents = self.network.get_parents(var)
            parent_values = {p: observed[p] for p in parents if p in observed}

            # For linear models: U = X - f(parents)
            # This is simplified; complex models need numerical methods
            if parent_values:
                predicted = self.structural_equations[var](
                    parent_values,
                    np.array([0])
                )[0]
                exogenous[var] = observed[var] - predicted
            else:
                exogenous[var] = observed[var]

        return exogenous

    def __repr__(self) -> str:
        """String representation."""
        n_equations = len(self.structural_equations)
        return f"StructuralCausalModel(equations={n_equations})"
