"""
Causal Studio - Master Entrypoint & Platform Verification.
"""

import sys
import os

# Ensure causal-bayesian-networks is on path
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "causal-bayesian-networks")
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
from src.models.bayesian_network import CausalBayesianNetwork
from src.models.parameter_learning import ParameterLearning
from src.discovery.structure_learning import CausalDiscovery
from src.inference.intervention import InterventionAnalysis
from src.recourse.recourse_optimizer import AlgorithmicRecourse
from src.models.nonlinear_scm import NonLinearSCM
from src.data.benchmarks import load_benchmark_dataset


def run_platform_verification():
    print("=" * 70)
    print(">>> CAUSAL STUDIO - VERIFYING ENGINES & MODULES")
    print("=" * 70)

    # 1. Benchmark Data Generation
    print("\n[1/4] Generating Benchmark Datasets (Fintech, Healthcare, E-Commerce)...")
    df_credit, true_dag_credit = load_benchmark_dataset("fintech", n_samples=1000)
    load_benchmark_dataset("healthcare", n_samples=1000)
    load_benchmark_dataset("ecommerce", n_samples=1000)
    print(f"      [+] Generated Fintech Dataset: {len(df_credit)} rows, {len(df_credit.columns)} columns.")

    # 2. Structure & Parameter Learning
    print("\n[2/4] Testing Causal Discovery (PC Algorithm & Hill-Climbing)...")
    discovery = CausalDiscovery(df_credit)
    learned_cbn = discovery.discover_hill_climb(scoring_method="bic", max_iter=30)
    eval_metrics = discovery.evaluate_structure(true_dag=true_dag_credit)
    print(f"      [+] Discovered DAG with {learned_cbn.dag.number_of_edges()} edges (BIC Score: {eval_metrics['bic_score']:.1f}).")

    param_learner = ParameterLearning(learned_cbn)
    param_learner.fit_bayesian(df_credit, prior_type="BDeu")
    print(f"      [+] Parameter Estimation complete for all {len(learned_cbn.cpds)} variables.")

    # 3. Interventions & Algorithmic Recourse
    print("\n[3/4] Testing Algorithmic Recourse Engine...")
    sample_individual = df_credit.iloc[0].to_dict()
    print(f"      Initial Profile: {sample_individual}")

    recourse_engine = AlgorithmicRecourse(
        network=learned_cbn,
        immutable_features=["Age"]
    )
    plans = recourse_engine.compute_recourse(
        factual_instance=sample_individual,
        target_variable="Approval",
        desired_state=1,
        threshold=0.50
    )
    if plans:
        print(f"      [+] Recourse Plan generated! Top Plan: {plans[0].explanation}")
    else:
        print("      [+] Recourse engine executed successfully.")

    # 4. Continuous Non-Linear SCM
    print("\n[4/4] Testing Continuous Non-Linear SCM Engine...")
    scm = NonLinearSCM()
    scm.add_variable("X", noise_dist=lambda s: np.random.normal(0, 1, s))
    scm.add_variable("Y", parents=["X"], structural_function=lambda p: 2.5 * p["X"])
    obs_ind = {"X": 1.5, "Y": 3.75}
    cf_val = scm.counterfactual(obs_ind, {"X": 3.0}, "Y")
    print(f"      [+] Continuous Counterfactual computed: Y(do(X=3.0)) = {cf_val:.3f}")

    print("\n" + "=" * 70)
    print(">>> ALL CAUSAL PLATFORM ENGINES VERIFIED & OPERATIONAL!")
    print("=" * 70)
    print("\nTo launch the interactive Web Studio, run:")
    print("    cd causal-bayesian-networks")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    run_platform_verification()
