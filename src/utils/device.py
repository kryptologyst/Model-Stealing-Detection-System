"""Device management utilities for model stealing detection."""

import os
import warnings
from typing import Optional

import torch


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    
    Priority: CUDA -> MPS -> CPU
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def set_deterministic(seed: int = 42) -> None:
    """
    Set deterministic behavior for reproducibility.
    
    Args:
        seed: Random seed for reproducibility
    """
    # Set Python random seed
    import random
    random.seed(seed)
    
    # Set NumPy random seed
    import numpy as np
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device_info() -> dict:
    """
    Get information about the current device.
    
    Returns:
        dict: Device information
    """
    device = get_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        })
    
    return info
