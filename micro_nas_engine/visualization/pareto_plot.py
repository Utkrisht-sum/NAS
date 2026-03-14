import matplotlib.pyplot as plt
from typing import List, Dict, Any
import os

from utils.logger import logger

def plot_pareto_frontier(all_architectures: List[Dict[str, Any]], pareto_frontier: List[Dict[str, Any]], save_path: str = "pareto_frontier.png"):
    """
    Plots the all architectures and highlights the Pareto frontier.
    X-axis: Compute Cost (Parameters)
    Y-axis: Validation Accuracy
    """
    try:
        # Extract data for all points
        all_costs = [arch["compute_cost"] for arch in all_architectures if arch["compute_cost"] < float('inf')]
        all_accs = [arch["accuracy"] for arch in all_architectures if arch["accuracy"] > -1.0]

        # Extract data for Pareto points
        pareto_costs = [arch["compute_cost"] for arch in pareto_frontier]
        pareto_accs = [arch["accuracy"] for arch in pareto_frontier]

        # Sort pareto frontier points by cost to draw a connected line
        pareto_points = sorted(zip(pareto_costs, pareto_accs), key=lambda x: x[0])
        sorted_pareto_costs = [p[0] for p in pareto_points]
        sorted_pareto_accs = [p[1] for p in pareto_points]

        plt.figure(figsize=(10, 6))

        # Plot all architectures (grey dots)
        plt.scatter(all_costs, all_accs, c='grey', alpha=0.5, label='Evaluated Architectures')

        # Plot Pareto frontier (red dots and line)
        plt.plot(sorted_pareto_costs, sorted_pareto_accs, 'r-', marker='o', label='Pareto Frontier')

        plt.title('Neural Architecture Search - Pareto Frontier')
        plt.xlabel('Compute Cost (Parameters)')
        plt.ylabel('Validation Accuracy')
        plt.xscale('log') # Use log scale for parameters as they can vary wildly
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Pareto frontier plot to {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate Pareto plot: {e}")
