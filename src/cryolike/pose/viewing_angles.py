import torch
from typing import Optional
from dataclasses import dataclass

from cryolike.grid import SphereShell
from cryolike.util import get_float_dtype, PrecisionLevel

@dataclass
class ViewingAngles:
    """Class storing the viewing angles, with weights, for a particular template/image stack.

    Viewing angles define the orientation of the particle relative to the imaging device. Thus,
    a ViewingAngles object is usually tied to a particular stack of observed images (Images)
    or reference images (Templates), with one value for each member for each image in the stack.
    However, if the ViewingAngles members are unit length in the highest dimension (i.e.
    len(azimus[0] == 1) then the angles can be broadcast to a whole image stack.

    Attributes:
        azimus (torch.Tensor): Azimuthal angle for each image
        polars (torch.Tensor): Polar angle for each image
        gammas (torch.Tensor): Gamma angle (grayscale correction factor) for each image
        weights_viewing (torch.Tensor): Weights to apply to each angle, per-image
        n_angles (int): Number of angles in the stack. For the ViewingAngles object
            to be paired with a Templates or Images object, the number of angles
            must match the number of images in that object, or be 1 (indicating that
            the angles will be broadcast over each image). The outermost length
            of each of the four tensors in the class must match the number of angles.
    """
    azimus: torch.Tensor
    polars: torch.Tensor
    gammas: torch.Tensor = torch.Tensor([])
    weights_viewing: torch.Tensor = torch.Tensor([])

    @property
    def n_angles(self) -> int:
        return self.azimus.shape[0]
    
    @property
    def precision(self) -> PrecisionLevel:
        return PrecisionLevel.SINGLE if self.azimus.dtype == torch.float32 else PrecisionLevel.DOUBLE

    def __post_init__(self):
        if self.gammas.numel() == 0:
            self.gammas = torch.zeros_like(self.azimus)
        if self.weights_viewing.numel() == 0:
            self.weights_viewing = torch.ones_like(self.azimus) / self.n_angles
        assert self.azimus.dtype == self.polars.dtype == self.gammas.dtype == self.weights_viewing.dtype, \
            "all tensors must have the same float precision"
        assert self.azimus.shape == self.polars.shape == self.gammas.shape == self.weights_viewing.shape, \
            "all tensors must have the same shape"

    @classmethod
    def from_viewing_distance(cls, viewing_distance: float, precision: Optional[PrecisionLevel] = None) -> "ViewingAngles":
        """Constructs a set of viewing angles from a regular viewing distance.

        Args:
            viewing_distance (float): The desired difference between two angles

        Returns:
            ViewingAngles: The set of viewing angles computed from this viewing distance
        """
        viewing_shell = SphereShell(radius=1.0, dist_eq=viewing_distance, uniform_azimuthal_sampling=False, store_cartesian_points=False)
        gammas = torch.zeros_like(viewing_shell.azimu_points)
        obj = cls(
            azimus=viewing_shell.azimu_points, 
            polars=viewing_shell.polar_points, 
            gammas=gammas, 
            weights_viewing=viewing_shell.weight_points
        )
        if precision is not None:
            obj = obj.to(dtype=get_float_dtype(precision))
        return obj

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        for attr in self.__dict__:
            _obj = getattr(self, attr)
            if isinstance(_obj, torch.Tensor):
                if dtype is not None:
                    _obj = _obj.to(dtype=dtype)
                if device is not None:
                    _obj = _obj.to(device=device)
                setattr(self, attr, _obj)
        return self
                

