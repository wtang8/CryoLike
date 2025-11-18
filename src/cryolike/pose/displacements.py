import torch
from cryolike.grid import UniformPolarGrid


class Displacements2D:

    x_displacements_angstrom: torch.Tensor
    y_displacements_angstrom: torch.Tensor
    box_size_angstrom: float

    @property
    def n_displacements(self):
        return self.x_displacements_angstrom.shape[0]

    @property
    def xy_displacements_angstrom(self):
        return torch.stack((self.x_displacements_angstrom, self.y_displacements_angstrom), dim=-1)

    def __init__(self,
        x_displacements_angstrom: torch.Tensor,
        y_displacements_angstrom: torch.Tensor,
        box_size_angstrom: float
    ):
        self.x_displacements_angstrom = x_displacements_angstrom
        self.y_displacements_angstrom = y_displacements_angstrom
        self.box_size_angstrom = box_size_angstrom

        assert self.x_displacements_angstrom.ndim == 1, "x_displacements must be a 1D array"
        assert self.y_displacements_angstrom.ndim == 1, "y_displacements must be a 1D array"
        assert self.x_displacements_angstrom.shape == self.y_displacements_angstrom.shape, "x_displacements and y_displacements must have the same shape"
        assert self.x_displacements_angstrom.dtype == self.y_displacements_angstrom.dtype, "x_displacements and y_displacements must have the same dtype"

    def to(self, dtype: torch.dtype, device: str | torch.device):
        self.x_displacements_angstrom = self.x_displacements_angstrom.to(dtype=dtype, device=device)
        self.y_displacements_angstrom = self.y_displacements_angstrom.to(dtype=dtype, device=device)
        return self
    
    def kernel(self, polar_grid: UniformPolarGrid):
        return polar_grid.get_fourier_translation_kernel(
            x_displacements_angstrom=self.x_displacements_angstrom,
            y_displacements_angstrom=self.y_displacements_angstrom,
            box_size=self.box_size_angstrom
        )
    
    @classmethod
    def sample_grid(cls, max_displacements_angstrom: float, n_samples_per_axis: int, box_size_angstrom: float):
        from cryolike.grid import SquaredCartesianGrid2D
        grid = SquaredCartesianGrid2D(box_size=max_displacements_angstrom * 2.0, n_pixels=n_samples_per_axis, endpoint=True)
        instance = cls(
            x_displacements_angstrom=grid.x_pixels.flatten(),
            y_displacements_angstrom=grid.y_pixels.flatten(),
            box_size_angstrom=box_size_angstrom
        )
        return instance

