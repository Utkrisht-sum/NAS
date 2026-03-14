from typing import List, Tuple, Dict, Any

def get_pareto_frontier(architectures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes the Pareto frontier of candidate architectures.
    We want to MAXIMIZE accuracy and MINIMIZE compute cost.
    """
    pareto_frontier = []

    # Sort architectures by compute cost (ascending)
    sorted_architectures = sorted(architectures, key=lambda x: x["compute_cost"])

    current_max_accuracy = -float('inf')

    for arch in sorted_architectures:
        # If this architecture has a strictly greater accuracy than the current max
        # seen for its compute cost or lower, it belongs on the frontier.
        if arch["accuracy"] > current_max_accuracy:
            pareto_frontier.append(arch)
            current_max_accuracy = arch["accuracy"]

    return pareto_frontier
