"""
Example: Simple medical treatment decision analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bayesian_network import CausalBayesianNetwork
from src.models.structural_causal_model import StructuralCausalModel
from src.inference.counterfactual import CounterfactualInference
from src.inference.intervention import InterventionAnalysis
from pgmpy.factors.discrete import TabularCPD


def main():
    """Run a simple medical treatment example."""

    print("=" * 60)
    print("Medical Treatment Decision Analysis Example")
    print("=" * 60)

    # Create network
    cbn = CausalBayesianNetwork()
    cbn.add_nodes(['Disease', 'Treatment', 'Recovery'])
    cbn.add_edges([
        ('Disease', 'Treatment'),
        ('Disease', 'Recovery'),
        ('Treatment', 'Recovery')
    ])

    # Define CPDs
    cpd_disease = TabularCPD('Disease', 2, [[0.7], [0.3]])

    cpd_treatment = TabularCPD('Treatment', 2,
                               [[0.9, 0.1],
                                [0.1, 0.9]],
                               evidence=['Disease'],
                               evidence_card=[2])

    cpd_recovery = TabularCPD('Recovery', 2,
                              [[0.95, 0.8, 0.5, 0.1],
                               [0.05, 0.2, 0.5, 0.9]],
                              evidence=['Disease', 'Treatment'],
                              evidence_card=[2, 2])

    cbn.set_cpd('Disease', cpd_disease)
    cbn.set_cpd('Treatment', cpd_treatment)
    cbn.set_cpd('Recovery', cpd_recovery)
    cbn.build_model()

    print(f"\nNetwork: {cbn}")
    print(f"Valid: {cbn.validate()}\n")

    # Intervention analysis
    print("-" * 60)
    print("INTERVENTION ANALYSIS")
    print("-" * 60)

    intervention = InterventionAnalysis(cbn)
    ate = intervention.average_treatment_effect('Treatment', 'Recovery')

    print(f"Average Treatment Effect: {ate:.3f}")
    print(f"Interpretation: Treatment increases recovery probability")
    print(f"                by {ate*100:.1f} percentage points\n")

    # Do-calculus queries
    p_recover_treat = intervention.do_intervention({'Treatment': 1}, 'Recovery')
    p_recover_no_treat = intervention.do_intervention({'Treatment': 0}, 'Recovery')

    print("P(Recovery | do(Treatment=1)):")
    for outcome, prob in p_recover_treat.items():
        print(f"  Recovery={outcome}: {prob:.3f}")

    print("\nP(Recovery | do(Treatment=0)):")
    for outcome, prob in p_recover_no_treat.items():
        print(f"  Recovery={outcome}: {prob:.3f}")

    print("\n" + "=" * 60)
    print("CONCLUSION: Treatment is beneficial for recovery")
    print("=" * 60)


if __name__ == "__main__":
    main()
