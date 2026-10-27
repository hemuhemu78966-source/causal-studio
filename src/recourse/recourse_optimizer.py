"""
Algorithmic Recourse and Actionable Decision Optimization Engine.

Solves the inverse problem: Given an individual's factual profile and an unfavorable outcome,
find the minimum-cost causal intervention set do(X* = x*) that guarantees a favorable outcome
subject to feasibility, actionability, and directional constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
import itertools
import networkx as nx
import numpy as np

from ..models.bayesian_network import CausalBayesianNetwork
from ..inference.intervention import InterventionAnalysis


@dataclass
class RecoursePlan:
    """Represents a computed recourse intervention plan."""
    interventions: Dict[str, Any]
    target_probability: float
    baseline_probability: float
    total_cost: float
    feature_changes: Dict[str, Dict[str, Any]]
    is_feasible: bool = True
    explanation: str = ""


class AlgorithmicRecourse:
    """
    Computes optimal, actionable causal recourse for individuals and business decisions.
    """

    def __init__(self,
                 network: CausalBayesianNetwork,
                 immutable_features: Optional[List[str]] = None,
                 action_costs: Optional[Dict[str, float]] = None,
                 directional_constraints: Optional[Dict[str, str]] = None,
                 feature_domains: Optional[Dict[str, List[Any]]] = None):
        """
        Initialize Algorithmic Recourse engine.

        Args:
            network: CausalBayesianNetwork model with fitted CPDs.
            immutable_features: List of variables that cannot be intervened on (e.g., Age, Gender).
            action_costs: Weights w_i specifying relative effort/cost to alter feature i.
            directional_constraints: Dict mapping variable to 'increasing' or 'decreasing' only.
            feature_domains: Dict mapping variable to allowed valid discrete states.
        """
        self.network = network
        self.intervention_engine = InterventionAnalysis(network)
        self.immutable_features = set(immutable_features or [])
        self.action_costs = action_costs or {}
        self.directional_constraints = directional_constraints or {}
        self.feature_domains = feature_domains or {}

        # Auto-populate feature domains from network if not provided
        for node in network.dag.nodes():
            if node not in self.feature_domains and node in network.cpds:
                cpd = network.cpds[node]
                card = cpd.variable_card
                if hasattr(cpd, 'state_names') and cpd.state_names and node in cpd.state_names:
                    self.feature_domains[node] = cpd.state_names[node]
                else:
                    self.feature_domains[node] = list(range(card))

    def compute_recourse(self,
                        factual_instance: Dict[str, Any],
                        target_variable: str,
                        desired_state: Any = 1,
                        threshold: float = 0.5,
                        max_interventions: int = 3,
                        top_k: int = 3) -> List[RecoursePlan]:
        """
        Find top-k minimum-cost recourse plans that achieve P(target = desired_state | do(X*)) >= threshold.

        Args:
            factual_instance: The individual's current observed state.
            target_variable: Name of outcome variable (e.g., 'Loan_Approved', 'Churn').
            desired_state: Target value to achieve (e.g., 1 or 'Approved').
            threshold: Minimum acceptable probability of achieving the desired state.
            max_interventions: Maximum number of simultaneous features to change (sparsity constraint).
            top_k: Number of alternative recourse options to return.

        Returns:
            List of RecoursePlan objects sorted from lowest cost to highest cost.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if any(w <= 0 for w in self.action_costs.values()):
            raise ValueError("Action costs must be positive")
        ancestors = nx.ancestors(self.network.dag, target_variable)
        actionable_nodes = sorted(node for node in ancestors
                                  if node not in self.immutable_features and node in factual_instance)
        affected = set(actionable_nodes)
        for node in actionable_nodes:
            affected.update(nx.descendants(self.network.dag, node))
        # Use the same pre-action context for every plan. Never fix mediators
        # or outcome descendants at their factual values after an action.
        context = {k: v for k, v in factual_instance.items()
                   if k not in affected and k != target_variable
                   and k not in nx.descendants(self.network.dag, target_variable)}
        baseline_actions = {k: factual_instance[k] for k in actionable_nodes}
        baseline_dist = self.intervention_engine.do_intervention(
            baseline_actions, target_variable, evidence=context)
        baseline_prob = float(baseline_dist.get(str(desired_state), 0.0))
        if baseline_prob >= threshold:
            return [RecoursePlan({}, baseline_prob, baseline_prob, 0.0, {},
                                 explanation="The current action settings already meet the model threshold; no change is needed.")]

        search_size = sum(sum(int(np.prod([max(0, len(self.feature_domains[n])-1) for n in subset]))
                              for subset in itertools.combinations(actionable_nodes, k))
                          for k in range(1, min(max_interventions, len(actionable_nodes))+1))
        if search_size > 3000:
            raise ValueError("This search exceeds 3,000 scenarios. Reduce maximum changes or lock more features.")
        valid_plans: List[RecoursePlan] = []

        # Search over candidate action subsets up to max_interventions
        for k in range(1, min(max_interventions + 1, len(actionable_nodes) + 1)):
            for subset in itertools.combinations(actionable_nodes, k):
                # Candidate values for each variable in the subset
                candidate_values_per_var = []
                for var in subset:
                    current_val = factual_instance[var]
                    domain = self.feature_domains.get(var, [0, 1])
                    valid_vals = []
                    for v in domain:
                        if v == current_val:
                            continue  # no change
                        # Check directional constraint
                        direction = self.directional_constraints.get(var, None)
                        if direction == "increasing" and v <= current_val:
                            continue
                        if direction == "decreasing" and v >= current_val:
                            continue
                        valid_vals.append(v)
                    candidate_values_per_var.append(valid_vals)

                # Skip if any variable has no valid moves
                if any(len(vals) == 0 for vals in candidate_values_per_var):
                    continue

                for value_combo in itertools.product(*candidate_values_per_var):
                    intervention_dict = dict(zip(subset, value_combo))

                    # Only changed features are forced; their descendants propagate.
                    result_dist = self.intervention_engine.do_intervention(
                        interventions=intervention_dict, target=target_variable,
                        evidence=context)
                    prob = float(result_dist.get(str(desired_state), 0.0))

                    reference_dist = self.intervention_engine.do_intervention(
                        {k: factual_instance[k] for k in intervention_dict},
                        target_variable, evidence=context)
                    plan_baseline = float(reference_dist.get(str(desired_state), 0.0))
                    if prob >= threshold and prob > plan_baseline + 1e-9:
                        # Compute recourse cost
                        cost = self._calculate_cost(factual_instance, intervention_dict)
                        feature_changes = {
                            var: {
                                "from": factual_instance[var],
                                "to": new_val,
                                "delta": (new_val - factual_instance[var]) if isinstance(new_val, (int, float)) and isinstance(factual_instance[var], (int, float)) else None
                            }
                            for var, new_val in intervention_dict.items()
                        }

                        explanation = self._format_explanation(feature_changes, target_variable, desired_state, prob, plan_baseline)

                        plan = RecoursePlan(
                            interventions=intervention_dict,
                            target_probability=prob,
                            baseline_probability=plan_baseline,
                            total_cost=cost,
                            feature_changes=feature_changes,
                            is_feasible=True,
                            explanation=explanation
                        )
                        valid_plans.append(plan)

        # Sort plans by lowest cost, then highest probability
        valid_plans.sort(key=lambda p: (p.total_cost, -p.target_probability))
        return valid_plans[:top_k]

    def _calculate_cost(self, factual: Dict[str, Any], intervention: Dict[str, Any]) -> float:
        """Compute weighted distance/effort cost for proposed interventions."""
        total_cost = 0.0
        for var, new_val in intervention.items():
            curr_val = factual[var]
            weight = self.action_costs.get(var, 1.0)

            # If numerical, calculate normalized absolute shift
            if isinstance(curr_val, (int, float)) and isinstance(new_val, (int, float)):
                domain = self.feature_domains.get(var, [0, 1])
                spread = max(domain) - min(domain) if len(domain) > 1 and max(domain) != min(domain) else 1.0
                dist = abs(new_val - curr_val) / spread
            else:
                dist = 1.0  # categorical state shift penalty

            total_cost += weight * dist

        return round(float(total_cost), 4)

    def _format_explanation(self,
                            changes: Dict[str, Dict[str, Any]],
                            target: str,
                            desired_state: Any,
                            prob: float,
                            baseline_prob: float) -> str:
        """Generate human-readable prescriptive explanation for decision makers."""
        actions = []
        for var, details in changes.items():
            actions.append(f"change {var} from '{details['from']}' to '{details['to']}'")
        actions_str = ", ".join(actions)
        pct_increase = (prob - baseline_prob) * 100
        return (f"To achieve '{target}={desired_state}' with model probability {prob*100:.1f}% "
                f"({pct_increase:+.1f} percentage points versus current settings), {actions_str}.")
