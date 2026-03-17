import torch
import psutil
from accelerate import Accelerator

MAX_VRAM_MB = 2500  # 2.5 GB

def get_accelerator():
    """
    Initializes and returns a HuggingFace Accelerator instance
    configured for memory constraints.
    """
    # Mixed precision to save memory
    accelerator = Accelerator(mixed_precision="fp16")
    return accelerator

def get_hardware_info():
    """
    Returns memory info to monitor RAM and VRAM usage.
    """
    info = {
        "ram_used_gb": psutil.virtual_memory().used / (1024 ** 3),
        "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "cpu_count": psutil.cpu_count(logical=True),
    }
    info["cuda_available"] = torch.cuda.is_available()

    if info["cuda_available"]:
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
        info["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    return info

def optimize_memory():
    """
    Clears PyTorch CUDA cache to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device(fallback_to_cpu=False):
    """
    Returns the appropriate device, with support for forced CPU fallback.
    """
    if fallback_to_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
