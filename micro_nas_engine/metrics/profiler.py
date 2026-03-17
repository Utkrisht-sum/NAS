import torch
import torch.nn as nn
from micro_nas_engine.utils.hardware import MAX_VRAM_MB

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model):
    """
    Rough estimation: FLOPs ≈ 2 * parameters.
    """
    return 2 * count_parameters(model)

def estimate_compute_cost(model, task_type, arch_config):
    """
    Approximation for compute cost: C(A) ≈ Σ (n(l−1) × n(l))
    """
    cost = 0
    if task_type == "Tabular":
        sizes = arch_config.get("hidden_sizes", [])
        if sizes:
            last_size = 100
            for size in sizes:
                cost += last_size * size
                last_size = size
            cost += last_size * 10
    else:
        cost = count_parameters(model)

    return cost

def estimate_memory_mb(model, input_shape, batch_size=32):
    """
    Estimates the memory footprint of a model in MB, including parameters and activations.
    Assuming fp32 (4 bytes per element) for worst-case, though we use fp16 in training.
    """
    param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    buffer_size = sum(b.numel() for b in model.buffers()) * 4 / (1024 ** 2)

    # Rough estimate of activation memory based on input shape and parameters
    # Very crude approximation: Activation memory ≈ 2 * param size * batch_size for deep nets
    activation_size = param_size * 2 * batch_size

    # Total estimated memory (Parameters + Gradients + Activations)
    total_mem_mb = param_size * 2 + buffer_size + activation_size

    return total_mem_mb

def is_architecture_memory_safe(model, input_shape, batch_size=32):
    """
    Checks if the architecture is safe to load and train on the current hardware setup.
    """
    est_mem = estimate_memory_mb(model, input_shape, batch_size)
    if est_mem > MAX_VRAM_MB:
        return False, est_mem
    return True, est_mem

def estimate_latency(model, input_shape, device="cpu"):
    """
    Provides a rough latency score proxy.
    In a real scenario, this would run a dummy forward pass and measure time.
    Here we proxy it using FLOPs and parameter counts.
    """
    # Simple proxy: parameters * a small constant
    return count_parameters(model) * 1e-8
