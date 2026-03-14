import os
import torch

class Config:
    # Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
    DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
    PRETRAIN_DATASETS_DIR = os.path.join(DATASETS_DIR, "pretraining")

    # Knowledge Files
    MEMORY_FILE = os.path.join(KNOWLEDGE_DIR, "architecture_memory.json")

    # NAS Defaults
    DEFAULT_POPULATION_SIZE = 10
    DEFAULT_GENERATIONS = 5
    DEFAULT_COMPUTE_BUDGET = 5_000_000  # FLOPs or Params

    # Fitness Default Weights
    DEFAULT_ALPHA = 1.0  # Accuracy weight
    DEFAULT_BETA = 1.0   # Compute penalty weight

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Search Space Bounds (Sensible defaults to prevent OOM)
    MLP_MAX_LAYERS = 5
    MLP_MAX_HIDDEN = 512

    CNN_MAX_LAYERS = 5
    CNN_MAX_FILTERS = 256

    # Training Defaults
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCHS = 10
    DEFAULT_LEARNING_RATE = 0.001

# Create required directories on load
os.makedirs(Config.KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(Config.PRETRAIN_DATASETS_DIR, exist_ok=True)
