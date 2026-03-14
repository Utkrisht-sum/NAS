from typing import List, Dict, Any, Union

def create_mlp_architecture(depth: int, hidden_units: List[int], activation: str, dropout: float) -> Dict[str, Any]:
    """
    Encodes an MLP architecture.
    A = (d, h1, h2, ..., a, p)
    """
    assert depth == len(hidden_units), "Depth must match number of hidden layers"
    return {
        "type": "mlp",
        "depth": depth,
        "hidden_units": hidden_units,
        "activation": activation,
        "dropout": dropout
    }

def create_cnn_architecture(depth: int, filters: List[int], kernel_sizes: List[int],
                            hidden_units: List[int], activation: str, dropout: float) -> Dict[str, Any]:
    """
    Encodes a CNN architecture.
    A = (d, f1, k1, f2, k2, ..., h1, h2, ..., a, p)
    """
    assert depth == len(filters) == len(kernel_sizes), "Depth must match number of conv layers"
    return {
        "type": "cnn",
        "depth": depth,
        "filters": filters,
        "kernel_sizes": kernel_sizes,
        "hidden_units": hidden_units,
        "activation": activation,
        "dropout": dropout
    }

def validate_architecture(arch: Dict[str, Any]) -> bool:
    """Validates the structure of an encoded architecture."""
    if "type" not in arch:
        return False

    if arch["type"] == "mlp":
        required = ["depth", "hidden_units", "activation", "dropout"]
        return all(k in arch for k in required) and len(arch["hidden_units"]) == arch["depth"]

    elif arch["type"] == "cnn":
        required = ["depth", "filters", "kernel_sizes", "hidden_units", "activation", "dropout"]
        return all(k in arch for k in required) and len(arch["filters"]) == arch["depth"] == len(arch["kernel_sizes"])

    return False
