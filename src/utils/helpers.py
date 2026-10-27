"""
Utility functions for causal Bayesian networks.
"""

import numpy as np
from typing import Dict, List, Any
import json


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Normalize an array of probabilities to sum to 1.

    Args:
        probs: Array of probability values

    Returns:
        Normalized probability array
    """
    total = np.sum(probs)
    if total == 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


def entropy(probs: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a probability distribution.

    Args:
        probs: Probability distribution

    Returns:
        Entropy value in bits
    """
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate KL divergence between two distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        KL divergence D(P||Q)
    """
    # Avoid division by zero
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    return np.sum(p * np.log(p / q))


def mutual_information(joint_probs: np.ndarray,
                      marginal_x: np.ndarray,
                      marginal_y: np.ndarray) -> float:
    """
    Calculate mutual information between two variables.

    Args:
        joint_probs: Joint probability distribution P(X,Y)
        marginal_x: Marginal distribution P(X)
        marginal_y: Marginal distribution P(Y)

    Returns:
        Mutual information I(X;Y)
    """
    mi = 0.0
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):
            if joint_probs[i, j] > 0:
                mi += joint_probs[i, j] * np.log(
                    joint_probs[i, j] / (marginal_x[i] * marginal_y[j])
                )
    return mi


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save analysis results to a JSON file.

    Args:
        results: Dictionary of results
        filepath: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load analysis results from a JSON file.

    Args:
        filepath: Path to load file

    Returns:
        Dictionary of results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def bootstrap_confidence_interval(data: np.ndarray,
                                 statistic: callable,
                                 n_bootstrap: int = 1000,
                                 confidence: float = 0.95) -> tuple:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Sample data
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper


def compute_effect_size_ci(treatment_outcomes: np.ndarray,
                          control_outcomes: np.ndarray,
                          confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute treatment effect with confidence interval.

    Args:
        treatment_outcomes: Outcomes under treatment
        control_outcomes: Outcomes under control
        confidence: Confidence level

    Returns:
        Dictionary with effect, lower_ci, upper_ci
    """
    effect = np.mean(treatment_outcomes) - np.mean(control_outcomes)

    # Bootstrap CI for the difference
    combined = np.concatenate([treatment_outcomes, control_outcomes])
    n_treatment = len(treatment_outcomes)

    def effect_statistic(sample):
        return np.mean(sample[:n_treatment]) - np.mean(sample[n_treatment:])

    lower, upper = bootstrap_confidence_interval(
        combined, effect_statistic, confidence=confidence
    )

    return {
        'effect': effect,
        'lower_ci': lower,
        'upper_ci': upper,
        'confidence': confidence
    }
