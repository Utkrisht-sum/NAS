import torch.nn as nn

def estimate_compute_cost(model: nn.Module) -> float:
    """
    Estimates the compute cost based on parameter count.
    This serves as our C(A) metric.
    """
    return float(sum(p.numel() for p in model.parameters() if p.requires_grad))

def estimate_flops(model: nn.Module) -> float:
    """
    Approximates FLOPs based on parameter count.
    FLOPs ≈ 2 * Parameters
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return float(2 * params)
