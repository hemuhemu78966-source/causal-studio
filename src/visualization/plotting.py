"""
Visualization utilities for causal networks and counterfactual analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any
from ..models.bayesian_network import CausalBayesianNetwork


def plot_causal_graph(network: CausalBayesianNetwork,
                     figsize: tuple = (10, 8),
                     node_color: str = 'lightblue',
                     highlight_nodes: Optional[List[str]] = None,
                     save_path: Optional[str] = None) -> None:
    """
    Visualize the causal graph structure.

    Args:
        network: CausalBayesianNetwork to visualize
        figsize: Figure size
        node_color: Default node color
        highlight_nodes: Optional list of nodes to highlight
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)

    # Use hierarchical layout for better visualization
    pos = nx.spring_layout(network.dag, k=2, iterations=50)

    # Color nodes
    colors = []
    for node in network.dag.nodes():
        if highlight_nodes and node in highlight_nodes:
            colors.append('orange')
        else:
            colors.append(node_color)

    nx.draw(network.dag, pos,
            with_labels=True,
            node_color=colors,
            node_size=2000,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=2)

    plt.title('Causal Graph Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_intervention_comparison(results: Dict[str, Dict[str, float]],
                                title: str = 'Intervention Comparison',
                                save_path: Optional[str] = None) -> None:
    """
    Compare probability distributions under different interventions.

    Args:
        results: Dictionary mapping intervention descriptions to probability dicts
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(next(iter(results.values()))))
    width = 0.8 / len(results)

    for i, (intervention, probs) in enumerate(results.items()):
        values = list(probs.values())
        labels = list(probs.keys())
        offset = width * i - 0.4 + width/2
        ax.bar(x + offset, values, width, label=intervention)

    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_treatment_effects(effects: Dict[str, float],
                          title: str = 'Treatment Effects',
                          save_path: Optional[str] = None) -> None:
    """
    Visualize treatment effects across variables.

    Args:
        effects: Dictionary mapping variable names to effect sizes
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    variables = list(effects.keys())
    values = list(effects.values())
    colors = ['green' if v > 0 else 'red' for v in values]

    ax.barh(variables, values, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Effect Size', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_counterfactual_scenarios(scenarios: List[Dict[str, Any]],
                                 target: str,
                                 save_path: Optional[str] = None) -> None:
    """
    Visualize multiple counterfactual scenarios.

    Args:
        scenarios: List of scenario dicts with 'name' and 'outcome' keys
        target: Target variable name
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [s['name'] for s in scenarios]
    outcomes = [s['outcome'] for s in scenarios]

    ax.scatter(range(len(names)), outcomes, s=200, alpha=0.6, c='blue')
    ax.plot(range(len(names)), outcomes, 'b--', alpha=0.3)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(f'{target} Value', fontsize=12)
    ax.set_title('Counterfactual Scenario Analysis', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_probability_distributions(distributions: Dict[str, np.ndarray],
                                  labels: Optional[List[str]] = None,
                                  title: str = 'Probability Distributions',
                                  save_path: Optional[str] = None) -> None:
    """
    Plot multiple probability distributions for comparison.

    Args:
        distributions: Dictionary mapping names to probability arrays
        labels: Optional list of x-axis labels
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, probs in distributions.items():
        x = labels if labels else np.arange(len(probs))
        ax.plot(x, probs, marker='o', label=name, linewidth=2)

    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_decision_heatmap(decision_matrix: np.ndarray,
                           row_labels: List[str],
                           col_labels: List[str],
                           title: str = 'Decision Matrix',
                           save_path: Optional[str] = None) -> None:
    """
    Create a heatmap for decision analysis.

    Args:
        decision_matrix: 2D array of decision values
        row_labels: Labels for rows (e.g., states)
        col_labels: Labels for columns (e.g., actions)
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(decision_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                xticklabels=col_labels,
                yticklabels=row_labels,
                cbar_kws={'label': 'Expected Utility'},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
