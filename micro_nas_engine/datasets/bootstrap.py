import os
import threading
from torchvision import datasets, transforms
import torchvision

def download_datasets(base_path="micro_nas_engine/datasets/pretraining"):
    os.makedirs(base_path, exist_ok=True)
    transform = transforms.ToTensor()

    print("Bootstrapping datasets in background...")
    try:
        # MNIST
        datasets.MNIST(root=base_path, train=True, download=True, transform=transform)
        # CIFAR10
        datasets.CIFAR10(root=base_path, train=True, download=True, transform=transform)
        # Fashion-MNIST
        datasets.FashionMNIST(root=base_path, train=True, download=True, transform=transform)
        # For UCI Iris and Wine, scikit-learn fetches them easily if needed via datasets.load_iris()
        # but we download torchvision ones to pretraining dir first.
        print("Bootstrap complete.")
    except Exception as e:
        print(f"Bootstrap error: {e}")

def run_bootstrap_if_needed(memory_file="micro_nas_engine/knowledge/architecture_memory.json"):
    # If architecture memory doesn't exist or is empty, start bootstrap in a thread
    needs_bootstrap = False
    if not os.path.exists(memory_file):
        needs_bootstrap = True
    else:
        try:
            with open(memory_file, 'r') as f:
                content = f.read().strip()
                if not content or content == "[]":
                    needs_bootstrap = True
        except:
            pass

    if needs_bootstrap:
        thread = threading.Thread(target=download_datasets, daemon=True)
        thread.start()
