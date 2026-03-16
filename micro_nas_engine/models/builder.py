import torch
import torch.nn as nn

class DynamicMLP(nn.Module):
    def __init__(self, in_features, out_features, arch_config):
        super().__init__()
        layers = []
        last_size = in_features
        for size in arch_config["hidden_sizes"]:
            layers.append(nn.Linear(last_size, size))
            if arch_config["activation"] == 'ReLU':
                layers.append(nn.ReLU())
            elif arch_config["activation"] == 'Tanh':
                layers.append(nn.Tanh())
            elif arch_config["activation"] == 'SiLU':
                layers.append(nn.SiLU())
            last_size = size
        layers.append(nn.Linear(last_size, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DynamicCNN(nn.Module):
    def __init__(self, in_channels, num_classes, arch_config):
        super().__init__()
        from micro_nas_engine.models.search_space import OPS_CNN

        self.cells = nn.ModuleList()
        current_channels = in_channels
        out_channels = 16

        cells_config = arch_config.get("cells", [])
        for cell_config in cells_config:
            op_name = cell_config[0][0] if cell_config and isinstance(cell_config[0], tuple) and isinstance(cell_config[0][0], str) else 'conv_3x3'
            if op_name not in OPS_CNN: op_name = 'conv_3x3' # Safe fallback

            op = OPS_CNN[op_name](current_channels, out_channels, stride=2 if current_channels == in_channels else 1)
            self.cells.append(op)
            current_channels = out_channels
            out_channels *= 2

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DynamicRNN(nn.Module):
    def __init__(self, in_features, num_classes, arch_config):
        super().__init__()
        from micro_nas_engine.models.search_space import OPS_RNN

        rnn_type = arch_config.get("rnn_type", "lstm")
        hidden_size = arch_config.get("hidden_size", 64)
        num_layers = arch_config.get("num_layers", 1)

        self.rnn = OPS_RNN[rnn_type](in_features, hidden_size)
        # Assuming simple multi-layer stack for now if > 1
        if num_layers > 1:
            self.rnn = nn.Sequential(*[OPS_RNN[rnn_type](in_features if i==0 else hidden_size, hidden_size) for i in range(num_layers)])

        self.fc = nn.Linear(hidden_size, num_classes)
        self._rnn_type = rnn_type

    def forward(self, x):
        if isinstance(self.rnn, nn.Sequential):
            out = x
            for layer in self.rnn:
                out, _ = layer(out)
        else:
            out, _ = self.rnn(x)

        # Take last time step
        out = out[:, -1, :]
        return self.fc(out)

class DynamicTransformerLoRA(nn.Module):
    def __init__(self, num_classes, arch_config):
        super().__init__()
        # In a real environment we would load from transformers
        # from transformers import AutoModelForSequenceClassification
        # from peft import LoraConfig, get_peft_model

        # model = AutoModelForSequenceClassification.from_pretrained(arch_config["model_name"], num_labels=num_classes)
        # lora_config = LoraConfig(r=arch_config["lora_r"], lora_alpha=arch_config["lora_alpha"], target_modules=["q_lin", "k_lin", "v_lin"])
        # self.model = get_peft_model(model, lora_config)

        # Stub for local compilation without downloading huggingface models:
        self.dummy_fc = nn.Linear(100, num_classes)
        self.arch_config = arch_config

    def forward(self, x):
        # Dummy forward
        return self.dummy_fc(x.float())

def build_model(task_type, input_shape, num_classes, arch_config):
    if task_type == "Tabular":
        return DynamicMLP(input_shape, num_classes, arch_config)
    elif task_type == "Image":
        return DynamicCNN(input_shape[0] if isinstance(input_shape, tuple) else 3, num_classes, arch_config)
    elif task_type == "Time-series":
        return DynamicRNN(input_shape[-1] if isinstance(input_shape, tuple) else input_shape, num_classes, arch_config)
    elif task_type == "Text":
        return DynamicTransformerLoRA(num_classes, arch_config)

    # Fallback
    return DynamicMLP(10, num_classes, {"hidden_sizes": [10], "activation": "ReLU"})
