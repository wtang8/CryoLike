from dataclasses import dataclass, field
import torch
import numpy as np
from cryolike.util.typechecks import ensure_positive
from cryolike.util.precision import get_float_dtype

@dataclass
class SquaredCartesianGrid2D:
    n_pixels: int
    box_size: float
    endpoint: bool = False
    pixel_size: float = field(init=False)
    x_axis: torch.Tensor = field(init=False, repr=False)
    y_axis: torch.Tensor = field(init=False, repr=False)
    x_pixels: torch.Tensor = field(init=False, repr=False)
    y_pixels: torch.Tensor = field(init=False, repr=False)
    n_pixels_total: int = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize computed fields after dataclass initialization."""
        ensure_positive(self.n_pixels, "n_pixels")
        ensure_positive(self.box_size, "box_size")
        self.n_pixels_total = self.n_pixels * self.n_pixels
        
        _radius = 0.5 * self.box_size
        _x_axis = np.linspace(-_radius, _radius, self.n_pixels, endpoint=self.endpoint)
        _y_axis = np.linspace(-_radius, _radius, self.n_pixels, endpoint=self.endpoint)
        self.pixel_size = _x_axis[1] - _x_axis[0]
        _x_pixels, _y_pixels = np.meshgrid(_x_axis, _y_axis, indexing='ij')

        float_dtype = get_float_dtype()
        self.x_axis = torch.from_numpy(_x_axis).to(dtype=float_dtype)
        self.y_axis = torch.from_numpy(_y_axis).to(dtype=float_dtype)
        self.x_pixels = torch.from_numpy(_x_pixels).to(dtype=float_dtype)
        self.y_pixels = torch.from_numpy(_y_pixels).to(dtype=float_dtype)
    
    def __repr__(self) -> str:
        return (f"SquaredCartesianGrid2D(n_pixels={self.n_pixels}, "
                f"pixel_size={self.pixel_size}, box_size={self.box_size}, "
                f"endpoint={self.endpoint})")
    
    def to(self, dtype: torch.dtype, device: str | torch.device):
        self.x_axis = self.x_axis.to(dtype=dtype, device=device)
        self.y_axis = self.y_axis.to(dtype=dtype, device=device)
        self.x_pixels = self.x_pixels.to(dtype=dtype, device=device)
        self.y_pixels = self.y_pixels.to(dtype=dtype, device=device)
        return self
        