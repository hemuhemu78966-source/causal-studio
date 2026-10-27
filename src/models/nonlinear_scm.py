"""
Continuous and Non-Linear Structural Causal Model (SCM).

Supports continuous variables, Additive Noise Models (ANMs), non-linear functional
mechanisms, and continuous counterfactual reasoning via Abduction-Action-Prediction.
"""

import numpy as np
import networkx as nx
from typing import Dict, Callable, List, Optional, Any, Tuple, Union
import pandas as pd


class NonLinearSCM:
    """
    Non-Linear Structural Causal Model for continuous and mixed variables.
    
    Models variables as Additive Noise Models (ANM):
        X_j = f_j(Parents(X_j)) + U_j
    where U_j are independent exogenous noise terms.
    """

    def __init__(self, dag: Optional[nx.DiGraph] = None):
        """
        Initialize NonLinearSCM.

        Args:
            dag: Optional NetworkX DiGraph representing the causal structure.
        """
        self.dag = dag or nx.DiGraph()
        self.functions: Dict[str, Callable[[Dict[str, np.ndarray]], np.ndarray]] = {}
        self.noise_generators: Dict[str, Callable[[int], np.ndarray]] = {}
        self.fitted_models: Dict[str, Any] = {}

    def add_variable(self,
                     name: str,
                     parents: Optional[List[str]] = None,
                     structural_function: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]] = None,
                     noise_dist: Optional[Callable[[int], np.ndarray]] = None) -> None:
        """
        Add a continuous endogenous variable to the SCM.

        Args:
            name: Variable name.
            parents: List of parent variable names.
            structural_function: Function mapping parent values dictionary to predicted mean.
            noise_dist: Callable generating exogenous noise samples of size N.
        """
        self.dag.add_node(name)
        if parents:
            for p in parents:
                self.dag.add_edge(p, name)

        if not nx.is_directed_acyclic_graph(self.dag):
            self.dag.remove_node(name)
            raise ValueError(f"Adding variable '{name}' with parents {parents} creates a cycle in the DAG.")

        if structural_function is not None:
            self.functions[name] = structural_function
        else:
            # Default root/identity equation returns 0.0
            self.functions[name] = lambda p: 0.0

        if noise_dist is not None:
            self.noise_generators[name] = noise_dist
        else:
            # Standard Gaussian noise by default
            self.noise_generators[name] = lambda size=1: np.random.normal(loc=0.0, scale=1.0, size=size)

    def sample(self,
               n_samples: int = 1000,
               interventions: Optional[Dict[str, Union[float, np.ndarray]]] = None,
               exogenous_noise: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        Sample continuous data from the SCM under optional interventions.

        Args:
            n_samples: Number of observational or interventional samples to draw.
            interventions: Dict of do(X=x) interventions.
            exogenous_noise: Optional pre-fixed exogenous noise dictionary (used in counterfactuals).

        Returns:
            pandas DataFrame of sampled variables.
        """
        interventions = interventions or {}
        samples: Dict[str, np.ndarray] = {}

        # Generate or assign noise
        if exogenous_noise is None:
            noise = {
                node: self.noise_generators.get(node, lambda s: np.random.randn(s))(n_samples)
                for node in self.dag.nodes()
            }
        else:
            noise = exogenous_noise

        # Forward simulate in topological order
        topological_order = list(nx.topological_sort(self.dag))
        for var in topological_order:
            if var in interventions:
                int_val = interventions[var]
                if isinstance(int_val, (int, float)):
                    samples[var] = np.full(n_samples, float(int_val))
                else:
                    samples[var] = np.asarray(int_val)
            else:
                parents = list(self.dag.predecessors(var))
                parent_vals = {p: samples[p] for p in parents}
                f = self.functions.get(var, lambda p: np.zeros(n_samples))
                u = noise.get(var, np.zeros(n_samples))
                
                # ANM: X = f(parents) + U
                samples[var] = f(parent_vals) + u

        return pd.DataFrame(samples)

    def counterfactual(self,
                       factual_observation: Dict[str, float],
                       intervention: Dict[str, float],
                       target_variable: str) -> float:
        """
        Compute continuous counterfactual outcome:
        "What would target_variable have been, had intervention do(X=x) occurred,
        given what was factually observed for this individual?"

        Steps:
        1. Abduction: Calculate exact noise residual u_i = x_i - f_i(parents_obs)
        2. Action: Mutilate structural equation for intervened variables
        3. Prediction: Re-compute target under new interventions and abducted noise terms

        Args:
            factual_observation: Single individual's observed profile.
            intervention: Target intervention dictionary do(X = x).
            target_variable: Outcome variable to compute.

        Returns:
            Counterfactual value of target variable.
        """
        # Step 1: Abduction
        abducted_noise: Dict[str, np.ndarray] = {}
        topological_order = list(nx.topological_sort(self.dag))

        for var in topological_order:
            if var in factual_observation:
                parents = list(self.dag.predecessors(var))
                parent_vals = {p: np.array([factual_observation[p]]) for p in parents if p in factual_observation}
                f = self.functions.get(var, lambda p: np.array([0.0]))
                predicted_val = f(parent_vals)[0] if parents else 0.0
                obs_val = factual_observation[var]
                
                # Residual noise
                abducted_noise[var] = np.array([obs_val - predicted_val])
            else:
                abducted_noise[var] = np.array([0.0])

        # Step 2 & 3: Action and Prediction
        cf_df = self.sample(
            n_samples=1,
            interventions=intervention,
            exogenous_noise=abducted_noise
        )

        return float(cf_df[target_variable].iloc[0])

    def individual_treatment_effect(self,
                                    factual_observation: Dict[str, float],
                                    treatment_var: str,
                                    treatment_value: float,
                                    control_value: float,
                                    outcome_var: str) -> float:
        """
        Compute Individual Treatment Effect (ITE):
        ITE(i) = Y(T = treatment_val, u_i) - Y(T = control_val, u_i)
        """
        y_treat = self.counterfactual(factual_observation, {treatment_var: treatment_value}, outcome_var)
        y_ctrl = self.counterfactual(factual_observation, {treatment_var: control_value}, outcome_var)
        return y_treat - y_ctrl
