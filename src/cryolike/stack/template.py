# from typing import Literal, Optional, cast
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
# from importlib.util import find_spec
from typing import NamedTuple
# from math import ceil

from .image import FourierImages
from cryolike.grid import UniformPolarGrid
from cryolike.pose import ViewingAngles
from cryolike.model import AtomicModel
from cryolike.util import PrecisionLevel, get_device, get_float_dtype, get_complex_dtype


class AtomShape(Enum):
    GAUSSIAN = "gaussian"
    HARD_SPHERE = "hard_sphere"
    DEFAULT = GAUSSIAN


def _fourier_circles(
    thetas: torch.Tensor,  # inplane angle in radians
    polars: torch.Tensor,  # polars angle in radians
    azimus: torch.Tensor,  # azimusthal angle in radians
    gammas: torch.Tensor,  # inplane rotation in radians
) -> torch.Tensor:
    
    thetas = thetas[None,:] + gammas[:,None]
    
    cos_thetas = torch.cos(thetas)
    sin_thetas = torch.sin(thetas)

    cos_polars = torch.cos(polars)
    sin_polars = torch.sin(polars)
    cos_azimus = torch.cos(azimus)
    sin_azimus = torch.sin(azimus)

    cos_thetas_cos_polars = cos_thetas * cos_polars[:,None]
    sin_thetas_sin_azimus = sin_thetas * sin_azimus[:,None]
    sin_thetas_cos_azimus = sin_thetas * cos_azimus[:,None]

    x_template_points = cos_azimus[:,None] * cos_thetas_cos_polars - sin_thetas_sin_azimus
    y_template_points = sin_azimus[:,None] * cos_thetas_cos_polars + sin_thetas_cos_azimus
    z_template_points = - sin_polars[:,None] * cos_thetas
    xyz_template_points = torch.stack((x_template_points, y_template_points, z_template_points), dim = 2)
    
    return xyz_template_points


def _get_offset(polar_grid: UniformPolarGrid) -> torch.Tensor:
    return torch.sinc(2.0 * polar_grid.x_points) * torch.sinc(2.0 * polar_grid.y_points)


def make_uniform_hard_sphere(
    atomic_model: AtomicModel, 
    polar_grid: UniformPolarGrid, 
    viewing_angles: ViewingAngles, 
    box_size: float,
) -> torch.Tensor:
    atomic_coordinates_scaled = atomic_model.coordinates / box_size * 2.0 * (- 2.0 * np.pi)        
    atomic_radius_scaled = atomic_model.atom_radii / box_size * 2.0
    xyz_template_points = _fourier_circles(
        polar_grid.theta_shell,
        viewing_angles.polars,
        viewing_angles.azimus,
        viewing_angles.gammas,
    )
    kR = 2 * np.pi * polar_grid.radius_shells[:,None] * atomic_radius_scaled[None,:] ## (n_shells, n_atoms)
    kernelAtoms = (torch.sin(kR) - kR * torch.cos(kR)) * polar_grid.radius_shells.pow(-3)[:,None] / (8 * np.pi ** 2) * 3 / atomic_model.n_atoms ## (n_shells, n_atoms)
    kdotr = torch.einsum(
        'tnx,fax,r->ftrna',
        xyz_template_points, ## (n_templates, n_inplanes, 3)
        atomic_coordinates_scaled, ## (n_frames, n_atoms, 3)
        polar_grid.radius_shells
    ) ## (n_frames, n_templates, n_shells, n_inplanes n_atoms)
    exponent = torch.exp(1j * kdotr) * kernelAtoms[None,None,:,None,:] ## (n_frames, n_templates, n_shells, n_inplanes, n_atoms)
    templates_fourier = torch.sum(exponent, dim = 4) ## (n_frames, n_templates, n_shells, n_inplanes)
    return templates_fourier


def make_uniform_gaussian(
    atomic_model: AtomicModel, 
    polar_grid: UniformPolarGrid, 
    viewing_angles: ViewingAngles, 
    box_size: float
) -> torch.Tensor:
    atomic_coordinates_scaled = atomic_model.coordinates / box_size * 2.0 * (- 2.0 * np.pi)        
    atomic_radius_scaled = atomic_model.atom_radii / box_size * 2.0
    pi_atomic_radius_sq_times_two = 2.0 * (np.pi * atomic_radius_scaled) ** 2
    radius_shells_sq = polar_grid.radius_shells ** 2
    xyz_template_points = _fourier_circles(
        polar_grid.theta_shell,
        viewing_angles.polars,
        viewing_angles.azimus,
        viewing_angles.gammas,
    )
    log_norm = - 1.5 * np.log(2 * np.pi) - 3 * torch.log(atomic_radius_scaled) - np.log(atomic_model.n_atoms)
    gaussKernelAtoms = - radius_shells_sq[:,None] * pi_atomic_radius_sq_times_two[None,:] + log_norm[None,:]
    kdotr = torch.einsum(
        'tnx,fax,r->ftrna',
        xyz_template_points, ## (n_templates, n_inplanes, 3)
        atomic_coordinates_scaled, ## (n_frames, n_atoms, 3)
        polar_grid.radius_shells
    ) ## (n_frames, n_templates, n_shells, n_inplanes n_atoms)
    exponent = torch.exp(torch.complex(gaussKernelAtoms[None,:,None,:], kdotr)) ## (n_frames, n_templates, n_shells, n_inplanes, n_atoms)
    # offset = _get_offset(polar_grid) * torch.sum(torch.exp(log_norm))
    templates_fourier = torch.sum(exponent, dim = 4) #- offset[None,None,:,:]
    return templates_fourier


class Templates(FourierImages):
    """Class representing a collection of (Cartesian-space and/or Fourier-space) templates, with methods for manipulating them.
    
    Attributes:
        box_size (FloatArrayType): Size of the (Cartesian-space) viewing port
        phys_grid (CartesianGrid2D): A grid describing the physical space in which the (physical-representation) templates reside
        polar_grid (PolarGrid): A grid describing the polar space in which the Fourier templates reside
        templates_phys (torch.Tensor | None): Cartesian-space template images as pixel-value array of [template x X-index x Y-index]
        templates_fourier (torch.Tensor | None): Fourier-space template images as complex-valued array of [template x radius x angle]
        n_templates (int): Count of templates in the collection
        ctf (CTF | None): Contrast transfer function to be applied to templates, if any
        viewing_angles (ViewingAngles): Viewing angle which generated each template in the stack
        filename (str | None): If set, the name of the file from which the templates were loaded
    """
    viewing_angles: ViewingAngles
    n_frames: int

    def __init__(self, *, viewing_angles: ViewingAngles, n_frames: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.viewing_angles = viewing_angles
        self.n_frames = n_frames
        if (self.n_frames * self.viewing_angles.n_angles != self.n_images):
            raise ValueError("Number of viewing angles must match number of templates.")
    
    @classmethod
    def generate_from_positions(
        cls,
        atomic_model: AtomicModel,
        viewing_angles: ViewingAngles,
        polar_grid: UniformPolarGrid,
        box_size: float,
        compute_device: str | torch.device | None = None,
        output_device: str | torch.device | None = "cpu",
        atom_shape: AtomShape = AtomShape.DEFAULT,
        precision: PrecisionLevel = PrecisionLevel.SINGLE,
    ):
        compute_device = get_device(compute_device)
        output_device = get_device(output_device)
        if atomic_model.use_protein_residue_model:
            atom_shape = AtomShape.HARD_SPHERE
            print("Using protein residue model, using hard sphere.")
        if atom_shape == AtomShape.DEFAULT:
            atom_shape = AtomShape.HARD_SPHERE
            print("Atom shape not specified, using hard sphere.")
        float_type = get_float_dtype(precision)
        atomic_model = atomic_model.to(dtype=float_type, device=compute_device)
        viewing_angles = viewing_angles.to(dtype=float_type, device=compute_device)
        polar_grid = polar_grid.to(dtype=float_type, device=compute_device)
        
        if atom_shape == AtomShape.GAUSSIAN:
            templates_fourier = make_uniform_gaussian(
                atomic_model=atomic_model,
                polar_grid=polar_grid,
                viewing_angles=viewing_angles,
                box_size=box_size
            )
        elif atom_shape == AtomShape.HARD_SPHERE:
            templates_fourier = make_uniform_hard_sphere(
                atomic_model=atomic_model,
                polar_grid=polar_grid,
                viewing_angles=viewing_angles,
                box_size=box_size
            )
        
        # Reshape the output to combine frames and templates for the Templates object
        n_frames, n_templates, n_shells, n_inplanes = templates_fourier.shape
        templates_fourier = templates_fourier.reshape(n_frames * n_templates, n_shells, n_inplanes)
        return cls(
            images_fourier=templates_fourier.to(device=output_device),
            polar_grid=polar_grid,
            box_size=box_size, 
            viewing_angles=viewing_angles, 
            n_frames=n_frames
        )


#     # @classmethod
#     # def generate_from_physical_volume(
#     #     cls,
#     #     volume: Volume,
#     #     polar_grid: PolarGrid,
#     #     viewing_angles: ViewingAngles,
#     #     precision: Precision = Precision.DEFAULT,
#     #     device: torch.device | None = None,
#     #     output_device: str | torch.device | None = "cpu",
#     #     nufft_eps: float = 1.0e-12,
#     #     verbose: bool = False
#     # ):
#     #     if volume.density_physical is None:
#     #         raise ValueError("No physical volume found")
#     #     _device = get_device(device)
#     #     _output_device = get_device(output_device)
#     #     (torch_float_type, _, _) = precision.get_dtypes(default=Precision.SINGLE)
#     #     n_templates = viewing_angles.n_angles

#     #     volume.density_physical = volume.density_physical.to(_device)
#     #     fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, float_type=torch_float_type, device=_device)
#     #     templates_fourier = volume_phys_to_fourier_points(
#     #         volume = volume,
#     #         fourier_slices = fourier_slices,
#     #         eps = nufft_eps,
#     #         precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
#     #         input_device = _device,
#     #         output_device = _output_device,
#     #         verbose = verbose
#     #     )

#     #     origin = torch.tensor([0.0, 0.0, 0.0], dtype = torch_float_type, device=_device).unsqueeze(0)
#     #     centers = volume_phys_to_fourier_points(
#     #         volume = volume,
#     #         fourier_slices = origin,
#     #         eps = nufft_eps,
#     #         precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
#     #         input_device = _device,
#     #         output_device= _output_device,
#     #         verbose = verbose
#     #     )
#     #     offset = _get_offset(polar_grid=polar_grid, float_type=torch_float_type, device=_output_device)
#     #     print("device:", offset.device, centers.device)
#     #     offset = offset[None,:] * centers
#     #     templates_fourier -= offset

#     #     if polar_grid.uniform:
#     #         templates_fourier = templates_fourier.reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)
#     #     data = FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)

#     #     return cls(fourier_data=data, viewing_angles=viewing_angles)


#     # @classmethod
#     # def generate_from_function(
#     #     cls,
#     #     function: Callable[[torch.Tensor], torch.Tensor],
#     #     viewing_angles: ViewingAngles,
#     #     polar_grid: PolarGrid,
#     #     device: str | torch.device | None = None,
#     #     output_device: str | torch.device | None = "cpu",
#     #     precision: Precision = Precision.DEFAULT
#     # ):
#     #     _device = get_device(device)
#     #     _output_device = get_device(output_device)
#     #     _test_callable(fn=function, precision=precision, device=_device)
#     #     (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
#     #     fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, float_type=torch_float_type, device=_device)
#     #     templates_fourier = function(fourier_slices).reshape(viewing_angles.n_angles, polar_grid.n_shells, polar_grid.n_inplanes).to(_output_device)
#     #     data = FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)
#     #     return cls(fourier_data=data, viewing_angles=viewing_angles)


#     # def to_images(self) -> Images:
#     #     if getattr(self, "phys_grid", None) is not None and self.has_physical_images():
#     #         phys_data = PhysicalImages(self.images_phys.clone(), pixel_size=self.phys_grid.pixel_size)
#     #     elif getattr(self, "phys_grid", None) is not None:
#     #         phys_data = self.phys_grid
#     #     else:
#     #         phys_data = None

#     #     if getattr(self, "polar_grid", None) is not None and self.has_fourier_images():
#     #         fourier_data = FourierImages(self.images_fourier.clone(), self.polar_grid)
#     #     elif getattr(self, "polar_grid", None) is not None:
#     #         fourier_data = self.polar_grid
#     #     else:
#     #         fourier_data = None

#     #     return Images(phys_data, fourier_data, self.box_size, self.viewing_angles)



# def _get_fourier_slices(polar_grid: PolarGrid, viewing_angles: ViewingAngles, float_type: torch.dtype, device: torch.device) -> torch.Tensor:
#     """Returns a tensor representing the Cartesian values, in space, of the points on the polar grid.

#     Outer index is the image, middle index is linearized point index, inner index is 3 (for x, y, z).
#     Linearized point index is (# of points per shell) x (# of shells)

#     Args:
#         polar_grid (PolarGrid): Grid to compute Fourier slices for
#         viewing_angles (ViewingAngles): Viewing angles used to compute slices
#         float_type (torch.dtype): dtype identifying level of precision to use for computation
#         device (torch.device): Device on which to carry out computatino

#     Returns:
#         torch.Tensor: Tensor of reals, of [img x point x [x/y/z]] such that result[0,5,:] is a
#             3-vector of the x,y,z coordinates of the 6th grid point of the 1st image in the stack.
#     """
#     radius_shells = torch.tensor(polar_grid.radius_shells, dtype=float_type, device=device)
#     circles = _get_circles(viewing_angles, polar_grid, float_type=float_type, device=device)
#     if not polar_grid.uniform:
#         return circles * radius_shells[None,:,None]
#     fourier_slices = circles.unsqueeze(1) * radius_shells[None,:,None,None]
#     fourier_slices = fourier_slices.flatten(1, 2)
#     return fourier_slices


# def _test_callable(fn: Callable[[torch.Tensor], torch.Tensor], precision: Precision, device: torch.device):
#     if not callable(fn):
#         raise ValueError("Function must be callable.")
#     (float_type, complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
#     test_input = torch.randn(1, 2, 3, dtype=float_type, device=device)
#     test_output: torch.Tensor = fn(test_input)
#     if test_output.shape != test_input.shape[:-1]:
#         raise ValueError("Function must be a callable that takes a tensor of shape (n_templates, n_pixels, 3) and returns a tensor of shape (n_templates, n_pixels).")
#     if precision != Precision.DEFAULT:
#         if test_output.dtype != complex_type:
#             raise ValueError(f"You have requested to work in {precision.value} precision, but the supplied function " +
#                              f"returns {test_output.dtype}. Please ensure your function returns the appropriate type.")
