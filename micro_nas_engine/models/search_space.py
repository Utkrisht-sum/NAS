import random
import torch.nn as nn
from collections import OrderedDict

# CNN operations for image search space
OPS_CNN = {
    'conv_3x3': lambda in_C, out_C, stride: nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_C, out_C, 3, stride, 1, bias=False)),
        ('bn', nn.BatchNorm2d(out_C)),
        ('relu', nn.ReLU(inplace=True))
    ])),
    'conv_5x5': lambda in_C, out_C, stride: nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_C, out_C, 5, stride, 2, bias=False)),
        ('bn', nn.BatchNorm2d(out_C)),
        ('relu', nn.ReLU(inplace=True))
    ])),
    'max_pool_3x3': lambda in_C, out_C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'avg_pool_3x3': lambda in_C, out_C, stride: nn.AvgPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda in_C, out_C, stride: nn.Identity() if stride == 1 and in_C == out_C else nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_C, out_C, 1, stride, 0, bias=False)),
        ('bn', nn.BatchNorm2d(out_C))
    ])),
    'none': lambda in_C, out_C, stride: Zero(stride)
}

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class MixedOp(nn.Module):
    def __init__(self, in_C, out_C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS_CNN.keys():
            op = OPS_CNN[primitive](in_C, out_C, stride)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(out_C))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


def sample_architecture(search_space_type, config=None):
    """
    Randomly generates architecture configuration from the search space.
    """
    if search_space_type == "Image":
        num_cells = config.get("num_cells", 3) if config else 3
        num_nodes = config.get("num_nodes", 4) if config else 4

        # simplified representation: list of tuples (op_name, input_idx)
        arch = []
        for cell in range(num_cells):
            cell_arch = []
            for node in range(num_nodes):
                # Sample 2 inputs and their operations for each node
                for i in range(2):
                    op = random.choice(list(OPS_CNN.keys()))
                    in_idx = random.randint(0, node + 1)
                    cell_arch.append((op, in_idx))
            arch.append(cell_arch)
        return {"cells": arch}
    elif search_space_type == "Tabular":
        # MLP: num_layers, list of hidden_sizes, activation
        num_layers = random.randint(1, 5)
        hidden_sizes = [random.choice([16, 32, 64, 128, 256]) for _ in range(num_layers)]
        activation = random.choice(['ReLU', 'Tanh', 'SiLU'])
        return {"num_layers": num_layers, "hidden_sizes": hidden_sizes, "activation": activation}
    return None

# Time-Series Operations
OPS_RNN = {
    'lstm': lambda in_size, hidden_size: nn.LSTM(in_size, hidden_size, batch_first=True),
    'gru': lambda in_size, hidden_size: nn.GRU(in_size, hidden_size, batch_first=True),
    'rnn': lambda in_size, hidden_size: nn.RNN(in_size, hidden_size, batch_first=True)
}

# Expand sample_architecture
def sample_architecture(search_space_type, config=None):
    if search_space_type == "Image":
        num_cells = config.get("num_cells", 3) if config else 3
        num_nodes = config.get("num_nodes", 4) if config else 4

        arch = []
        for cell in range(num_cells):
            cell_arch = []
            for node in range(num_nodes):
                for i in range(2):
                    op = random.choice(list(OPS_CNN.keys()))
                    in_idx = random.randint(0, node + 1)
                    cell_arch.append((op, in_idx))
            arch.append(cell_arch)
        return {"cells": arch}
    elif search_space_type == "Tabular":
        num_layers = random.randint(1, 5)
        hidden_sizes = [random.choice([16, 32, 64, 128, 256]) for _ in range(num_layers)]
        activation = random.choice(['ReLU', 'Tanh', 'SiLU'])
        return {"num_layers": num_layers, "hidden_sizes": hidden_sizes, "activation": activation}
    elif search_space_type == "Time-series":
        rnn_type = random.choice(list(OPS_RNN.keys()))
        hidden_size = random.choice([32, 64, 128])
        num_layers = random.choice([1, 2, 3])
        return {"rnn_type": rnn_type, "hidden_size": hidden_size, "num_layers": num_layers}
    elif search_space_type == "Text":
        # Simulate simple transformer search space with LoRA Rank
        model_name = random.choice(["distilbert-base-uncased", "prajjwal1/bert-tiny"])
        lora_r = random.choice([4, 8, 16])
        lora_alpha = random.choice([16, 32])
        return {"model_name": model_name, "lora_r": lora_r, "lora_alpha": lora_alpha}
    return None
