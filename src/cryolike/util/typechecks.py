import torch
import numpy as np

def ensure_positive(x: float | int |  np.ndarray | torch.Tensor, desc: str):
    if type(x) in [float, int]:
        if (x <= 0.0):
            raise ValueError(f'{desc} must be positive, got {x})')
    else:
        failed = False
        if isinstance(x, torch.Tensor):
            failed = torch.any(x <= 0.0).item()
        else:
            failed = np.any(x <= 0.0)
        if failed:
            raise ValueError(f'{desc} must be positive, got {x})')