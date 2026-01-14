import torch
from importlib.util import find_spec
from typing import Optional

from cryolike.util import PrecisionLevel

def check_nufft_installed(dev: torch.device) -> None:
    """Check for the availability of NUFFT libraries."""
    if dev.type == 'cuda':
        spec = find_spec('cufinufft')
        if spec is None:
            raise Exception("CUDA is requested, but cufinufft is not installed.")
    else:
        spec = find_spec('finufft')
        if spec is None:
            raise Exception("CPU is requested, but finufft is not installed.")
    pass


def get_epsilon(precision: PrecisionLevel, requested: Optional[float]) -> float:
    """Get epsilon value for the specified precision."""
    if precision == PrecisionLevel.SINGLE:
        min_epsilon = 1e-6
    else:
        min_epsilon = 1e-12
    if requested is None:
        return min_epsilon
    if requested < min_epsilon:
        print(f"Requested epsilon {requested} is too small for precision {precision.name}, setting to {min_epsilon}.")
        return min_epsilon
    return requested