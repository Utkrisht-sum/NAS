import random
import torch.nn as nn
from collections import OrderedDict
import copy

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

# Time-Series Operations
OPS_RNN = {
    'lstm': lambda in_size, hidden_size: nn.LSTM(in_size, hidden_size, batch_first=True),
    'gru': lambda in_size, hidden_size: nn.GRU(in_size, hidden_size, batch_first=True),
    'rnn': lambda in_size, hidden_size: nn.RNN(in_size, hidden_size, batch_first=True)
}

def sample_architecture(search_space_type, config=None):
    if search_space_type == "Image":
        num_cells = config.get("num_cells", random.randint(2, 5)) if config else random.randint(2, 5)
        num_nodes = config.get("num_nodes", random.randint(2, 4)) if config else random.randint(2, 4)

        arch = []
        for cell in range(num_cells):
            cell_arch = []
            for node in range(num_nodes):
                for i in range(2):
                    op = random.choice(list(OPS_CNN.keys()))
                    in_idx = random.randint(0, node + 1)
                    cell_arch.append((op, in_idx))
            arch.append(cell_arch)
        return {"cells": arch, "num_cells": num_cells, "dna": f"C{num_cells}N{num_nodes}"}
    elif search_space_type == "Tabular":
        num_layers = random.randint(1, 5)
        hidden_sizes = [random.choice([16, 32, 64, 128, 256]) for _ in range(num_layers)]
        activation = random.choice(['ReLU', 'Tanh', 'SiLU'])
        return {"num_layers": num_layers, "hidden_sizes": hidden_sizes, "activation": activation, "dna": f"L{num_layers}S{sum(hidden_sizes)}A{activation}"}
    elif search_space_type == "Time-series":
        rnn_type = random.choice(list(OPS_RNN.keys()))
        hidden_size = random.choice([32, 64, 128])
        num_layers = random.choice([1, 2, 3])
        return {"rnn_type": rnn_type, "hidden_size": hidden_size, "num_layers": num_layers, "dna": f"R{rnn_type}H{hidden_size}L{num_layers}"}
    elif search_space_type == "Text":
        model_name = random.choice(["distilbert-base-uncased", "prajjwal1/bert-tiny"])
        lora_r = random.choice([4, 8, 16])
        lora_alpha = random.choice([16, 32])
        return {"model_name": model_name, "lora_r": lora_r, "lora_alpha": lora_alpha, "dna": f"M{model_name}R{lora_r}A{lora_alpha}"}
    return None

def validate_cnn_spatial_dimensions(arch_config, input_shape):
    """
    Validates if a CNN architecture results in valid spatial dimensions.
    H_out = (H - K + 2P)//S + 1
    If H_out <= 0 or W_out <= 0, architecture is invalid.
    """
    if len(input_shape) != 3:
        return True, arch_config # Not an image

    _, H, W = input_shape
    cells = arch_config.get("cells", [])

    current_H, current_W = H, W

    for cell_idx, cell in enumerate(cells):
        op_name = cell[0][0] if isinstance(cell[0], tuple) else 'conv_3x3'

        stride = 2 if cell_idx == 0 else 1
        K = 3 if '3x3' in op_name else 5 if '5x5' in op_name else 1
        P = 1 if K == 3 else 2 if K == 5 else 0

        current_H = (current_H - K + 2*P)//stride + 1
        current_W = (current_W - K + 2*P)//stride + 1

        if current_H <= 0 or current_W <= 0:
            return False, arch_config

    return True, arch_config

def repair_architecture(arch_config, task_type):
    """
    Instead of rejecting, repairs an architecture.
    """
    repaired = copy.deepcopy(arch_config)

    if task_type == "Image":
        # Repair invalid CNNs by trimming depth
        if "cells" in repaired and len(repaired["cells"]) > 1:
            repaired["cells"].pop()
            repaired["num_cells"] = len(repaired["cells"])
    elif task_type == "Tabular":
        # Repair MLP by halving sizes or reducing layers
        if "num_layers" in repaired and repaired["num_layers"] > 1:
            repaired["num_layers"] -= 1
            repaired["hidden_sizes"].pop()
        else:
            repaired["hidden_sizes"] = [max(8, s // 2) for s in repaired["hidden_sizes"]]

    # Update DNA
    if task_type == "Image":
        repaired["dna"] = f"C{len(repaired.get('cells', []))}N{len(repaired.get('cells', [[]])[0])//2 if repaired.get('cells') else 0}"
    elif task_type == "Tabular":
        repaired["dna"] = f"L{repaired.get('num_layers', 0)}S{sum(repaired.get('hidden_sizes', [0]))}A{repaired.get('activation', 'ReLU')}"

    return repaired
