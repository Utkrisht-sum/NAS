import os
import json
import torch
from micro_nas_engine.models.builder import build_model

def export_deployment_project(model, config, task_type, input_shape, num_classes, output_dir="trained_model"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # 2. Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "task_type": task_type,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "arch_config": config
        }, f, indent=4)

    # 3. Create predict.py
    predict_script = """import torch
import json
import sys
import numpy as np
from PIL import Image

# Import builder from the engine (assuming it's installed or available)
try:
    from micro_nas_engine.models.builder import build_model
except ImportError:
    print("Please ensure micro_nas_engine is in your PYTHONPATH.")
    sys.exit(1)

def load_model(model_dir="."):
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)

    model = build_model(
        config["task_type"],
        config["input_shape"],
        config["num_classes"],
        config["arch_config"]
    )
    model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    model.eval()
    return model, config

def predict(input_data_path, model_dir="."):
    model, config = load_model(model_dir)

    # Dummy preprocessing based on type
    if config["task_type"] == "Image":
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((config["input_shape"][1], config["input_shape"][2])),
                transforms.ToTensor()
            ])
            img = Image.open(input_data_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
        except Exception as e:
            print(f"Error loading image: {e}")
            return
    else:
        # Tabular/Text dummy
        print("Tabular/Text prediction not fully implemented in this template.")
        return

    with torch.no_grad():
        out = model(tensor)
        preds = torch.softmax(out, dim=1)
        cls = torch.argmax(preds, dim=1).item()

    print(f"Predicted class index: {cls} with confidence {preds[0][cls].item():.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_data_path>")
        sys.exit(1)
    predict(sys.argv[1])
"""
    with open(os.path.join(output_dir, "predict.py"), "w") as f:
        f.write(predict_script)

    # 4. Create requirements.txt
    reqs = """torch
torchvision
numpy
pillow
"""
    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write(reqs)

    # 5. Create README.md
    readme = """# Deployed Model

## Usage
1. Install requirements:
   `pip install -r requirements.txt`

2. Run prediction:
   `python predict.py <path_to_input>`
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)

    print(f"Deployment project exported to {output_dir}/")
