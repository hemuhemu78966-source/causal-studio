"""
Counterfactual inference for decision analysis.

Implements methods for answering counterfactual queries:
"What would have happened if...?"
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pgmpy.inference import VariableElimination
from ..models.bayesian_network import CausalBayesianNetwork
from ..models.structural_causal_model import StructuralCausalModel


class CounterfactualInference:
    """
    Performs counterfactual inference on causal Bayesian networks.

    Supports both probability-based (using Bayesian networks) and
    SCM-based counterfactual reasoning.
    """

    def __init__(self,
                 network: Optional[CausalBayesianNetwork] = None,
                 scm: Optional[StructuralCausalModel] = None):
        """
        Initialize counterfactual inference engine.

        Args:
            network: CausalBayesianNetwork for probabilistic inference
            scm: StructuralCausalModel for functional-based inference
        """
        self.network = network
        self.scm = scm
        self.inference_engine = None

        if network and network.model:
            self.inference_engine = VariableElimination(network.model)

    def query_counterfactual(self,
                            observed: Dict[str, Any],
                            intervention: Dict[str, Any],
                            target: str,
                            method: str = 'scm') -> Any:
        """
        Answer a counterfactual query.

        Query format: P(Y=y | do(X=x), Z=z) where:
        - Y is the target variable
        - do(X=x) is an intervention
        - Z=z are observed variables

        Args:
            observed: Dictionary of observed variable values
            intervention: Dictionary of intervention values
            target: Target variable to predict
            method: 'scm' for structural approach, 'bounds' for probabilistic bounds

        Returns:
            Counterfactual outcome (value or probability distribution)
        """
        if method == 'scm':
            if self.scm is None:
                raise ValueError("SCM required for structural counterfactual inference")
            return self.scm.counterfactual(observed, intervention, target)
        elif method == 'bounds':
            return self._compute_counterfactual_bounds(observed, intervention, target)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_counterfactual_bounds(self,
                                      observed: Dict[str, Any],
                                      intervention: Dict[str, Any],
                                      target: str) -> Tuple[float, float]:
        """
        Compute bounds on counterfactual probabilities.

        When the full SCM is unknown, we can still bound counterfactual
        probabilities using observed distributions.

        Args:
            observed: Observed values
            intervention: Intervention values
            target: Target variable

        Returns:
            (lower_bound, upper_bound) on counterfactual probability
        """
        if self.inference_engine is None:
            raise ValueError("Bayesian network model required for bound computation")

        # Compute P(target | do(intervention))
        intervention_query = self.inference_engine.query(
            variables=[target],
            evidence=intervention,
            do=intervention.keys()
        )

        # Compute P(target | observed)
        observed_query = self.inference_engine.query(
            variables=[target],
            evidence=observed
        )

        # Tighten bounds using monotonicity and consistency
        # This is a simplified version; full bounds require more analysis
        lower = max(0, intervention_query.values[1] -
                   (1 - observed_query.values[1]))
        upper = min(1, intervention_query.values[1] +
                   (1 - observed_query.values[1]))

        return lower, upper

    def probability_of_necessity(self,
                                cause: str,
                                effect: str,
                                observed: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute Probability of Necessity (PN).

        PN = P(effect would not have occurred without cause | effect and cause occurred)

        Args:
            cause: Causal variable
            effect: Effect variable
            observed: Optional additional observations

        Returns:
            PN value between 0 and 1
        """
        if self.scm is None:
            raise ValueError("SCM required for PN computation")

        observed = observed or {}
        observed[cause] = 1
        observed[effect] = 1

        # Counterfactual: would effect=0 if we set cause=0?
        cf_result = self.scm.counterfactual(
            observed=observed,
            intervention={cause: 0},
            target=effect
        )

        # PN = P(effect_cf = 0 | cause=1, effect=1)
        return float(cf_result == 0)

    def probability_of_sufficiency(self,
                                  cause: str,
                                  effect: str,
                                  observed: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute Probability of Sufficiency (PS).

        PS = P(effect would have occurred with cause | no effect and no cause)

        Args:
            cause: Causal variable
            effect: Effect variable
            observed: Optional additional observations

        Returns:
            PS value between 0 and 1
        """
        if self.scm is None:
            raise ValueError("SCM required for PS computation")

        observed = observed or {}
        observed[cause] = 0
        observed[effect] = 0

        # Counterfactual: would effect=1 if we set cause=1?
        cf_result = self.scm.counterfactual(
            observed=observed,
            intervention={cause: 1},
            target=effect
        )

        # PS = P(effect_cf = 1 | cause=0, effect=0)
        return float(cf_result == 1)

    def explain_outcome(self,
                       observed: Dict[str, Any],
                       target: str,
                       variables_to_explain: List[str]) -> Dict[str, float]:
        """
        Explain an observed outcome by computing counterfactual effects.

        For each variable, compute what the target would have been if that
        variable had been different.

        Args:
            observed: Observed values
            target: Target variable to explain
            variables_to_explain: List of potential causal variables

        Returns:
            Dictionary mapping variables to their counterfactual effects
        """
        if self.scm is None:
            raise ValueError("SCM required for outcome explanation")

        effects = {}
        baseline = observed[target]

        for var in variables_to_explain:
            # Compute counterfactual with opposite value
            intervention_value = 1 - observed[var] if observed[var] in [0, 1] else 0
            cf_result = self.scm.counterfactual(
                observed=observed,
                intervention={var: intervention_value},
                target=target
            )

            # Effect = change in target
            effects[var] = cf_result - baseline

        return effects
