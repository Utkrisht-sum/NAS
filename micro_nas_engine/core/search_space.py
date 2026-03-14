import random
from typing import Dict, Any, List

from utils.config import Config

class SearchSpace:
    """
    Defines the boundaries and valid choices for generating and mutating neural architectures.
    """
    def __init__(self, task_type: str):
        self.task_type = task_type

        # Valid Activations
        self.activations = ["relu", "leakyrelu", "tanh", "gelu"]

        # Dropout Ranges
        self.dropout_min = 0.0
        self.dropout_max = 0.5

        # MLP Bounds
        self.mlp_depth_range = (1, Config.MLP_MAX_LAYERS)
        self.mlp_hidden_range = (16, Config.MLP_MAX_HIDDEN)

        # CNN Bounds
        self.cnn_depth_range = (1, Config.CNN_MAX_LAYERS)
        self.cnn_filter_range = (16, Config.CNN_MAX_FILTERS)
        self.cnn_kernel_choices = [3, 5]

    def sample_random_architecture(self) -> Dict[str, Any]:
        """Generates a random valid architecture based on task type."""
        if self.task_type == "tabular":
            return self._sample_mlp()
        elif self.task_type == "image":
            return self._sample_cnn()
        else:
            raise ValueError(f"Unsupported task type for search space: {self.task_type}")

    def _sample_mlp(self) -> Dict[str, Any]:
        depth = random.randint(*self.mlp_depth_range)
        hidden_units = [random.choice([16, 32, 64, 128, 256, 512]) for _ in range(depth)]

        return {
            "type": "mlp",
            "depth": depth,
            "hidden_units": hidden_units,
            "activation": random.choice(self.activations),
            "dropout": round(random.uniform(self.dropout_min, self.dropout_max), 2)
        }

    def _sample_cnn(self) -> Dict[str, Any]:
        depth = random.randint(*self.cnn_depth_range)
        filters = [random.choice([16, 32, 64, 128, 256]) for _ in range(depth)]
        kernel_sizes = [random.choice(self.cnn_kernel_choices) for _ in range(depth)]

        # Add 1 or 2 fully connected layers after CNN features
        fc_depth = random.randint(1, 2)
        hidden_units = [random.choice([32, 64, 128, 256]) for _ in range(fc_depth)]

        return {
            "type": "cnn",
            "depth": depth,
            "filters": filters,
            "kernel_sizes": kernel_sizes,
            "hidden_units": hidden_units,
            "activation": random.choice(self.activations),
            "dropout": round(random.uniform(self.dropout_min, self.dropout_max), 2)
        }
