import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class MLPBuilder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        layers = []
        in_features = input_dim

        activation_fn = self._get_activation(config["activation"])
        dropout_rate = config["dropout"]

        for hidden_size in config["hidden_units"]:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten if input is a higher-dimensional tensor
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def _get_activation(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        elif name == "leakyrelu":
            return nn.LeakyReLU
        elif name == "tanh":
            return nn.Tanh
        elif name == "gelu":
            return nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {name}")

class CNNBuilder(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], num_classes: int, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        channels, height, width = input_dim

        conv_layers = []
        in_channels = channels

        activation_fn = self._get_activation(config["activation"])

        # Build Convolutional Blocks
        for out_channels, kernel_size in zip(config["filters"], config["kernel_sizes"]):
            padding = kernel_size // 2
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            conv_layers.append(activation_fn())
            conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

            # Simple manual calculation to update spatial dimensions
            height = height // 2
            width = width // 2

            # Prevent dimension from becoming 0
            if height == 0 or width == 0:
                raise ValueError("Architecture produced 0 spatial dimensions. Invalid configuration.")

        self.features = nn.Sequential(*conv_layers)

        # Compute flattened size
        flattened_size = in_channels * height * width

        # Build Fully Connected Layers
        fc_layers = []
        in_features = flattened_size
        dropout_rate = config["dropout"]

        for hidden_size in config["hidden_units"]:
            fc_layers.append(nn.Linear(in_features, hidden_size))
            fc_layers.append(activation_fn())
            if dropout_rate > 0.0:
                fc_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        fc_layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _get_activation(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        elif name == "leakyrelu":
            return nn.LeakyReLU
        elif name == "tanh":
            return nn.Tanh
        elif name == "gelu":
            return nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {name}")

def build_model(arch: Dict[str, Any], input_dim: Any, num_classes: int) -> nn.Module:
    """Factory function to build a PyTorch module from an architecture dictionary."""
    if arch["type"] == "mlp":
        # Handle input_dim if it comes in as a tuple like (1, 28, 28)
        if isinstance(input_dim, tuple):
            import math
            flat_dim = math.prod(input_dim)
        else:
            flat_dim = input_dim
        return MLPBuilder(flat_dim, num_classes, arch)

    elif arch["type"] == "cnn":
        # Ensure input_dim is a tuple of (C, H, W)
        if not isinstance(input_dim, tuple) or len(input_dim) != 3:
            raise ValueError("CNN requires a tuple input dimension of format (C, H, W)")
        return CNNBuilder(input_dim, num_classes, arch)

    else:
        raise ValueError(f"Unsupported architecture type: {arch['type']}")
