"""
Comprehensive Unit and Integration Tests for Causal Discovery, Recourse & Continuous SCM.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.models.bayesian_network import CausalBayesianNetwork
from src.models.parameter_learning import ParameterLearning
from src.models.nonlinear_scm import NonLinearSCM
from src.discovery.structure_learning import CausalDiscovery
from src.inference.intervention import InterventionAnalysis
from src.recourse.recourse_optimizer import AlgorithmicRecourse
from src.data.benchmarks import generate_credit_risk_dataset, generate_healthcare_dataset


class TestCausalDiscovery:
    """Tests for structure learning and discovery algorithms."""

    @pytest.fixture
    def sample_data(self):
        df, true_dag = generate_credit_risk_dataset(n_samples=500, seed=123)
        return df, true_dag

    def test_hill_climb_discovery(self, sample_data):
        df, true_dag = sample_data
        discovery = CausalDiscovery(df)
        cbn = discovery.discover_hill_climb(scoring_method="bic", max_iter=50)
        assert isinstance(cbn, CausalBayesianNetwork)
        assert cbn.dag.number_of_nodes() == len(df.columns)
        assert cbn.dag.number_of_edges() > 0
        assert nx.is_directed_acyclic_graph(cbn.dag)

    def test_evaluate_structure(self, sample_data):
        df, true_dag = sample_data
        discovery = CausalDiscovery(df)
        cbn = discovery.discover_hill_climb(scoring_method="bic", max_iter=20)
        metrics = discovery.evaluate_structure(true_dag=true_dag)
        assert "bic_score" in metrics
        assert "structural_hamming_distance" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


class TestParameterLearning:
    """Tests for parameter estimation (MLE & Bayesian)."""

    def test_bayesian_and_mle_estimation(self):
        df, _ = generate_healthcare_dataset(n_samples=500, seed=42)
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(list(df.columns))
        cbn.add_edges([
            ('Age', 'Disease_Severity'),
            ('Disease_Severity', 'Treatment'),
            ('Treatment', 'Recovery')
        ])

        learner = ParameterLearning(cbn)
        fitted_cbn = learner.fit_bayesian(df, prior_type="BDeu")
        assert len(fitted_cbn.cpds) == len(df.columns)
        assert fitted_cbn.validate()

        summary = learner.get_cpd_summary()
        assert "Recovery" in summary
        assert "Treatment" in summary


class TestAlgorithmicRecourse:
    """Tests for algorithmic recourse optimizer."""

    def test_recourse_generation(self):
        df, _ = generate_credit_risk_dataset(n_samples=600, seed=42)
        cbn = CausalBayesianNetwork()
        cbn.add_nodes(list(df.columns))
        cbn.add_edges([
            ('Income', 'Savings'),
            ('Savings', 'Credit_Score'),
            ('Credit_Score', 'Approval'),
            ('Savings', 'Approval')
        ])
        learner = ParameterLearning(cbn)
        learner.fit_bayesian(df)

        factual = {
            'Age': 0,
            'Income': 0,
            'Savings': 0,
            'Loan_Amount': 2,
            'Credit_Score': 0,
            'Installment_Rate': 1,
            'Approval': 0
        }

        recourse = AlgorithmicRecourse(
            network=cbn,
            immutable_features=['Age']
        )

        plans = recourse.compute_recourse(
            factual_instance=factual,
            target_variable='Approval',
            desired_state=1,
            threshold=0.50,
            top_k=2
        )

        assert len(plans) > 0
        top_plan = plans[0]
        assert top_plan.target_probability >= 0.50
        assert 'Age' not in top_plan.interventions  # Immutable check


class TestNonLinearSCM:
    """Tests for continuous non-linear structural causal models."""

    def test_continuous_simulation_and_counterfactual(self):
        scm = NonLinearSCM()
        scm.add_variable("X", noise_dist=lambda s: np.random.normal(0, 1, s))
        scm.add_variable(
            "Y",
            parents=["X"],
            structural_function=lambda p: 3.0 * p["X"] + 2.0,
            noise_dist=lambda s: np.random.normal(0, 0.5, s)
        )

        # Sampling
        samples = scm.sample(n_samples=200)
        assert len(samples) == 200
        assert "X" in samples.columns
        assert "Y" in samples.columns

        # Counterfactual: observed X=2, Y=8.2 (noise residual was 0.2)
        # What if X was 4? Expected Y = 3*4 + 2 + 0.2 = 14.2
        factual = {"X": 2.0, "Y": 8.2}
        cf_y = scm.counterfactual(factual, {"X": 4.0}, "Y")
        assert np.isclose(cf_y, 14.2, atol=1e-3)
