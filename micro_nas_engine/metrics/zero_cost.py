import torch
import torch.nn as nn

@torch.no_grad()
def calculate_synflow(model, dataloader, device):
    """
    Approximates the SynFlow zero-cost proxy.
    Original SynFlow computes the L1 norm of the gradients with an all-ones input
    without data. This is a simplified version just checking forward pass activation sparsity
    or variance as a fast proxy for this demo.
    """
    model.eval()
    model.to(device)

    # We'll just do a very basic forward pass with ones and sum the output norms
    # to roughly rank architectures in zero-cost manner

    try:
        inputs, _ = next(iter(dataloader))
        ones_input = torch.ones_like(inputs).to(device)

        # Turn parameters to absolute values
        params_orig = []
        for p in model.parameters():
            params_orig.append(p.data.clone())
            p.data = p.data.abs()

        out = model(ones_input)
        score = torch.sum(out).item()

        # Restore parameters
        for p, orig in zip(model.parameters(), params_orig):
            p.data = orig

        return score
    except Exception as e:
        return 0.0

@torch.no_grad()
def calculate_naswot(model, dataloader, device):
    """
    Approximates NASWOT (Neural Architecture Search without Training).
    Computes a score based on the correlation of activations for a batch of data.
    """
    model.eval()
    model.to(device)

    try:
        inputs, _ = next(iter(dataloader))
        inputs = inputs.to(device)

        # Forward pass
        out = model(inputs)

        # If output is flat (N, C) compute kernel matrix
        if len(out.shape) == 2:
            # Binary activation pattern
            binary_acts = (out > 0).float()
            # Hamming distance approximation
            K = torch.matmul(binary_acts, binary_acts.T)
            # Log determinant of kernel matrix
            score = torch.slogdet(K + 1e-5 * torch.eye(K.shape[0]).to(device))[1].item()
            return score
        return 0.0
    except Exception as e:
        return 0.0

def filter_architectures(models_with_configs, dataloader, device, keep_ratio=0.1):
    """
    Applies zero-cost filters to eliminate weak architectures.
    Returns the top keep_ratio proportion.
    """
    scores = []
    for model, config in models_with_configs:
        # Use NASWOT as primary proxy
        score = calculate_naswot(model, dataloader, device)
        scores.append((score, model, config))

    # Sort descending
    scores.sort(key=lambda x: x[0], reverse=True)
    keep_count = max(1, int(len(scores) * keep_ratio))

    return [(item[1], item[2]) for item in scores[:keep_count]]
