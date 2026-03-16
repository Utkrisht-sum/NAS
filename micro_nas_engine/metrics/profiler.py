import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model):
    """
    Very rough estimation: FLOPs ≈ 2 * parameters.
    """
    return 2 * count_parameters(model)

def estimate_compute_cost(model, task_type, arch_config):
    """
    Approximation for compute cost: C(A) ≈ Σ (n(l−1) × n(l))
    For MLPs: sum of products of adjacent layer sizes.
    For CNNs: approximated by parameters for now.
    """
    cost = 0
    if task_type == "Tabular":
        # MLP
        sizes = arch_config.get("hidden_sizes", [])
        if sizes:
            # Assume an input size of e.g. 100
            last_size = 100
            for size in sizes:
                cost += last_size * size
                last_size = size
            # Assume output size e.g. 10
            cost += last_size * 10
    else:
        # For CNNs use parameters as proxy
        cost = count_parameters(model)

    return cost
