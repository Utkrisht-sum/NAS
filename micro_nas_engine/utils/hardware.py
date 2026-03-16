import torch
import psutil
from accelerate import Accelerator

def get_accelerator():
    """
    Initializes and returns a HuggingFace Accelerator instance
    configured for our memory constraints (3GB VRAM).
    """
    # Use mixed precision (fp16) to save memory
    accelerator = Accelerator(mixed_precision="fp16")
    return accelerator

def get_hardware_info():
    """
    Returns memory info to monitor RAM and VRAM usage.
    """
    info = {
        "ram_used_gb": psutil.virtual_memory().used / (1024 ** 3),
        "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
    }
    if torch.cuda.is_available():
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
    return info

def optimize_memory():
    """
    Clears PyTorch CUDA cache to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
