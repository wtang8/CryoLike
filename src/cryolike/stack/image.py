from typing import Literal, Optional, cast
import numpy as np
import mrcfile
from math import ceil
import torch
from torch import device, tensor, Tensor
from dataclasses import dataclass

from cryolike.grids import (
    SquaredCartesianGrid2D,
    UniformPolarGrid,
)
# from cryolike.microscopy import (
#     CTF,
#     fourier_polar_to_cartesian_phys,
#     cartesian_phys_to_fourier_polar,
# )
from cryolike.util import (
    ensure_positive,
    PrecisionLevel,
    get_precision
    # Cartesian_grid_2d_descriptor,
    # ComplexArrayType,
    # FloatArrayType,
    # get_imgs_max,
    # get_device,
    # IntArrayType,
    # NormType,
    # Precision,
    # project_descriptor,
    # TargetType,
    # to_torch,
)
# from cryolike.metadata import (
#     ViewingAngles
# )


@dataclass
class PhysicalImages:
    """Class representing a collection of Physical images, with methods for manipulating them.

    Attributes:
    """
    phys_grid: SquaredCartesianGrid2D
    images_phys: Tensor
    filename: Optional[str] = None

    @property
    def n_images(self) -> int:
        return self.images_phys.shape[0]
    
    @property
    def n_pixels(self) -> int:
        return self.phys_grid.n_pixels
    
    @property
    def box_size(self) -> float:
        return self.phys_grid.box_size
    
    @property
    def pixel_size(self) -> float:
        return self.phys_grid.pixel_size

    def __post_init__(self):
        expected_shape = (self.n_images, self.n_pixels, self.n_pixels)
        if self.images_phys.shape != expected_shape:
            raise ValueError(f"Dimension mismatch: expected images shape {expected_shape} but got {self.images_phys.shape}")
        
    @classmethod
    def from_mrc(cls, filename: str, pixel_size: float, device: str | torch.device = 'cpu'):
        """Create a new set of physical images from an MRC file.

        Args:
            filename (str): Name of MRC file to load. Must end with a .mrc or .mrcs extension.
            pixel_size (float): Sizes of the image pixels in Angstroms. If set,
                will override the values in the MRC file; if unset, the file values will be used.
            device (str | torch.device, optional): Device to use for the resulting image Tensor. Defaults to 'cpu'.

        Raises:
            ValueError: If a non-MRC file extension is passed.
            ValueError: If the pixel_size is not set and the existing MRC file has non-positive pixel sizes.

        Returns:
            Images: An Images image collection, with the physical images and grid populated per the saved data.
        """
        if not (filename.endswith('.mrc') or filename.endswith('.mrcs')):
            raise ValueError("Invalid file format. Only .mrc or .mrcs files are supported.")
        
        images_phys = None
        with mrcfile.open(filename) as mrc:
            assert isinstance(mrc.data, np.ndarray)
            imgs_phys = mrc.data.copy()
            if len(imgs_phys.shape) == 2:
                imgs_phys = imgs_phys[None, :, :]
            ensure_positive(pixel_size, "pixel size")
            images_phys = torch.from_numpy(imgs_phys).to(device)
            if images_phys.shape[1] != images_phys.shape[2]:
                raise ValueError("Non-square images are not supported yet.")
        if images_phys is None:
            raise RuntimeError("Unreachable: images_phys is still None after loading MRC file.")
        
        n_pixels = images_phys.shape[1]
        box_size = n_pixels * pixel_size
        phys_grid = SquaredCartesianGrid2D(n_pixels, box_size, endpoint=False)
        im = cls(images_phys=images_phys, phys_grid=phys_grid)
        im.filename = filename
        return im
    
    def save_to_mrc(self, filename: str):
        """Write the physical images to an MRC file. Will overwrite any existing file with that name.

        Args:
            filename (str): Name of the file to save.
        """
        with mrcfile.new(filename, overwrite = True) as mrc:
            mrc.set_data(self.images_phys)
            mrc.voxel_size = (self.pixel_size, self.pixel_size, 1.0)

    def to(self, dtype: torch.dtype, device: str | torch.device) -> 'PhysicalImages':
        """Convert all tensors in this object to the specified data type and device."""
        self.phys_grid = self.phys_grid.to(dtype=dtype, device=device)
        self.images_phys = self.images_phys.to(dtype=dtype, device=device)
        return self
    
    def normalize_images(self, ord: int = 1, use_max: bool = False) -> Tensor:
        """Normalize the physical images in this collection.

        Args:
            ord (int, optional): Degree of norm to apply. Defaults to 1.
            use_max (bool, optional): Whether to use the max in place of an LP norm. Defaults to False.

        Returns:
            torch.Tensor: The norm applied to the images (which are modified in-place).
        """
        if use_max:
            self.images_phys -= torch.amin(self.images_phys, dim=(1,2), keepdim=True)
            self.images_phys /= torch.amax(self.images_phys, dim=(1,2), keepdim=True)
            return self.images_phys
        else:
            lpnorm = torch.norm(self.images_phys.view(self.n_images, -1), p=ord, dim=1)
            for i in range(len(self.images_phys.shape) - 1):
                lpnorm = torch.unsqueeze(lpnorm, dim = -1)
            self.images_phys /= lpnorm
            return self.images_phys        

    def select_images(self, indices: list[int] | np.ndarray | torch.Tensor):
        """Restrict this collection's images to the specified indices.

        Args:
            indices (_type_): The indices to keep.
        """
        self.images_phys = self.images_phys[indices]

    def downsample_images_phys(
        self,
        downsample_factor: int = 2,                
        type: Literal['mean'] | Literal['max'] = 'mean'
    ) -> 'PhysicalImages':
        """Downsample the physical images in this collection.

        Args:
            downsample_factor (int, optional): Factor by which to downsample the images. Defaults to 2.
            type (Literal['mean'] | Literal['max']): Whether to take mean or max of the pool for
                downsampling. Defaults to mean.

        Raises:
            ValueError: If physical images do not exist in this collection.
        """
        ensure_positive(downsample_factor, "downsample factor")
        if downsample_factor == 1:
            return self
        _images_phys = self.images_phys.unsqueeze(1)  # NOTE: need to add channel dimension for nn.functional pooling
        if type == 'mean':
            _images_phys = torch.nn.functional.avg_pool2d(_images_phys, downsample_factor, downsample_factor)
        elif type == 'max':
            _images_phys = torch.nn.functional.max_pool2d(_images_phys, downsample_factor, downsample_factor)
        _images_phys = _images_phys.squeeze(1)
        n_pixels = _images_phys.shape[1]
        _phys_grid = SquaredCartesianGrid2D(n_pixels=n_pixels, box_size=self.box_size, endpoint=False)
        im = PhysicalImages(images_phys=_images_phys, phys_grid=_phys_grid)
        return im
    
    def transform_to_fourier(
        self,
        polar_grid: UniformPolarGrid,
        precision: PrecisionLevel | None = None,
        device: str | device = 'cuda'
    ):
        """Transform the physical images in this collection to Fourier-space representation. Existing
        physical images are kept. The new Fourier-space images will be placed on the same device as the
        physical images.

        Args:
            polar_grid (UniformPolarGrid): The polar grid for the Fourier-space image
                representation
            nufft_eps (float, optional): Tolerance for non-uniform FFT. Defaults to 1e-12.
            precision (Precision, optional): Whether to use 64- or 128-bit representation.
                (Note that the Fourier-space representation uses complex numbers, so the bit
                counts are twice the corresponding precision for reals.) Defaults to
                Precision.DEFAULT, which matches the precision of the physical representation.
            use_cuda (bool, optional): Whether to use a cuda device. Defaults to True.

        Raises:
            ValueError: If no polar grid exists on the collection already, and none is passed.
        """
        if not torch.cuda.is_available():
            device = 'cpu'
        if precision is None:
            precision = get_precision()
        nufft_eps = 1e-12 if precision == PrecisionLevel.DOUBLE else 1e-5
        raise NotImplementedError("Fourier transform not implemented yet.")

        # self.images_fourier = cartesian_phys_to_fourier_polar(
        #     grid_cartesian_phys = self.phys_grid,
        #     grid_fourier_polar = self.polar_grid,
        #     images_phys = self.images_phys,
        #     eps = nufft_eps,
        #     precision = precision,
        #     device = device
        # )
        # self.images_fourier = self.images_fourier.reshape(self.n_images, self.polar_grid.n_shells, self.polar_grid.n_inplanes)
        # self.images_fourier.to(self.images_phys.device)


@dataclass
class FourierImages:
    """Class representing a collection of Fourier-space images, with methods for manipulating them."""

    polar_grid: UniformPolarGrid
    box_size: float
    images_fourier: Tensor

    @property
    def n_images(self) -> int:
        return self.images_fourier.shape[0]
    
    @property
    def n_shells(self) -> int:
        return self.polar_grid.n_shells
    
    @property
    def n_inplanes(self) -> float:
        return self.polar_grid.n_inplanes
    
    def __post_init__(self):
        expected_shape = (self.n_images, self.n_shells, self.n_inplanes)
        if self.images_fourier.shape != expected_shape:
            raise ValueError(f"Dimension mismatch: expected images shape {expected_shape} but got {self.images_fourier.shape}")
        
    def to(self, dtype: torch.dtype, device: str | torch.device) -> 'FourierImages':
        """Convert all tensors in this object to the specified data type and device."""
        self.polar_grid = self.polar_grid.to(dtype=dtype, device=device)
        self.images_fourier = self.images_fourier.to(dtype=dtype, device=device)
        return self

    


    # def transform_to_spatial(
    #     self,
    #     grid: CartesianGrid2D | Cartesian_grid_2d_descriptor | None = None,
    #     nufft_eps: float = 1e-12,
    #     precision: Precision = Precision.DEFAULT,
    #     max_to_transform: int = -1,
    #     device: str | device | None = None
    # ) -> torch.Tensor:
    #     """Transform the Fourier-space images in this collection to a Cartesian-space representation.
    #     Existing Fourier-space images are kept. The new images will be placed on the same device as the
    #     Fourier-space images.

    #     Args:
    #         grid (Optional[Cartesian_grid_descriptor], optional): Cartesian grid describing the new physical
    #             images. Defaults to None. If None is passed, the collection's existing grid will be used.
    #             If None is passed and no existing grid exists, this operation will fail.
    #         nufft_eps (float, optional): Tolerance for the non-uniform FFT. Defaults to 1e-12.
    #         precision (Precision, optional): Whether to use 32- or 64-bit representation.
    #             Defaults to Precision.DEFAULT (which matches the current Fourier precision).
    #         use_cuda (bool, optional): Whether to use a cuda device. Defaults to True.

    #     Raises:
    #         ValueError: If no existing Cartesian grid was set, and no new one was passed.
    #     """
    #     self._ensure_fourier_images()
    #     if grid is None and getattr(self, "phys_grid", None) is None:
    #         raise ValueError('No physical grid found, and physical grid parameters were not provided.')
    #     if grid is not None:
    #         self.phys_grid = CartesianGrid2D.from_descriptor(grid)
    #     persist_transformed = False
    #     if max_to_transform <= 0:
    #         if max_to_transform == -1:
    #             persist_transformed = True
    #         max_to_transform = self.n_images
    #     if not persist_transformed:
    #         print(f"Transforming only the first {max_to_transform} images, probably for testing or plotting. Transformed images will be returned but not persisted.")

    #     device = self.images_fourier.device if device is None else device
    #     _device = get_device(device)
    #     images_fourier = self.images_fourier[:max_to_transform]
    #     images_fourier = images_fourier.reshape(images_fourier.shape[0], -1)
    #     if precision == Precision.DEFAULT:
    #         precision = Precision.SINGLE if images_fourier.dtype == torch.complex64 else Precision.DOUBLE
    #     else:
    #         if (precision == Precision.SINGLE and images_fourier.dtype != torch.complex64) or \
    #            (precision == Precision.DOUBLE and images_fourier.dtype != torch.complex128):
    #             print("Precision %s provided, overriding the existing precision." % precision.value)
    #     images_phys = fourier_polar_to_cartesian_phys(
    #         grid_fourier_polar = self.polar_grid,
    #         grid_cartesian_phys = self.phys_grid,
    #         image_polar = images_fourier,
    #         eps = nufft_eps,
    #         precision = precision,
    #         device = _device
    #     ).real
    #     if persist_transformed:
    #         self.images_phys = images_phys.to(device)
    #     return images_phys

    
    # def center_physical_image_signal(self, norm_type: NormType = NormType.MAX):
    #     """Converts pixel values to 0-mean and unit norm representation.

    #     Args:
    #         norm_type (NormType, optional): Whether to use max-norm or standard-deviation
    #             norm. Defaults to NormType.MAX.

    #     Raises:
    #         ValueError: If physical images do not exist in this collection.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor]: The computed mean and norm of the images.
    #     """
    #     self._ensure_phys_images()
    #     mean = torch.mean(self.images_phys, dim = (1,2), keepdim=True)
    #     self.images_phys -= mean
    #     if norm_type == NormType.MAX:
    #         norm = torch.amax(torch.abs(self.images_phys), dim = (1,2), keepdim = True)
    #     elif norm_type == NormType.STD:
    #         norm = torch.std(self.images_phys, dim = (1,2), keepdim = True)
    #     else:
    #         raise ValueError("Unreachable: unsupported norm type.")
    #     self.images_phys /= norm
    #     return mean, norm


    # def apply_ctf(self, ctf: CTF):
    #     """Applies a contrast transfer function to the Fourier-space images.

    #     Args:
    #         ctf (CTF): CTF to apply.

    #     Raises:
    #         NotImplementedError: If the Fourier-space images are using a non-uniform
    #             polar grid.

    #     Returns:
    #         torch.Tensor: The updated images (which will also be persisted to the collection).
    #     """
    #     self._ensure_fourier_images()
    #     if not self.polar_grid.uniform:
    #         raise NotImplementedError("Non-uniform Fourier images not implemented yet.")
    #     self.ctf = ctf
    #     self.images_fourier = ctf.apply(self.images_fourier)
    #     return self.images_fourier


    # def add_noise_phys(self, snr: float | FloatArrayType | torch.Tensor = 1.0):
    #     """Add random ((0,1) Gaussian) noise to the Cartesian-space images in the collection.

    #     Args:
    #         snr (float | FloatArrayType | torch.Tensor, optional): Signal-to-noise ratio. Defaults to 1.0.

    #     Raises:
    #         ValueError: If Cartesian-space images do not exist.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor]: The updated images (as also updated in-place), and the
    #             noise applied to them. 
    #     """
    #     ensure_positive(snr, "signal-to-noise ratio")
    #     if not self.has_physical_images():
    #         raise ValueError("Atempting to add physical noise, but physical images are not set.")
    #     device = self.images_phys.device
    #     # if not torch.cuda.is_available():
    #     #     device = 'cpu'
    #     if isinstance(snr, np.ndarray):
    #         snr = torch.tensor(snr).to(device)
    #     power_image = torch.mean(torch.abs(self.images_phys) ** 2, dim = (1,2))
    #     sigma_noise = torch.sqrt(power_image / snr).unsqueeze(1).unsqueeze(2)
    #     noise = sigma_noise * torch.randn_like(self.images_phys)
    #     self.images_phys = self.images_phys + noise
    #     return self.images_phys, sigma_noise.flatten().cpu().numpy()


    # def add_noise_fourier(self, snr: float | FloatArrayType | torch.Tensor = 1.0):
    #     """Add random ((0,1) complex normal) noise to Fourier-space images in the collection.

    #     Args:
    #         snr (float | FloatArrayType | torch.Tensor, optional): Signal-to-noise ratio. Defaults to 1.0.

    #     Raises:
    #         ValueError: If Fourier-space images do not exist.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor]: The updated images (as also updated in-place), and the
    #             noise applied to them. 
    #     """
    #     ensure_positive(snr, "signal-to-noise ratio")
    #     if self.images_fourier.shape[0] == 0:
    #         raise ValueError("Attempting to add fourier noise, but fourier images are not set.")
    #     device = self.images_fourier.device
    #     if isinstance(snr, np.ndarray):
    #         snr = torch.tensor(snr).to(device)
    #     power_image = self.polar_grid.integrate(self.images_fourier.abs().pow(2))
    #     power_image = power_image.unsqueeze(1).unsqueeze(2)
    #     sigma_noise = torch.sqrt(power_image / snr).unsqueeze(1).unsqueeze(2)
    #     noise = torch.randn_like(self.images_fourier) * sigma_noise
    #     self.images_fourier = self.images_fourier + noise
    #     return self.images_fourier, sigma_noise.flatten().cpu().numpy()


    # def set_displacement_grid(
    #     self,
    #     max_displacement_pixels: float,
    #     n_displacements_x: int,
    #     n_displacements_y: int,
    #     pixel_size_angstrom: float | None = None,
    # ):
    #     """Sets the displacement grid (the set of displacements to search
    #     in cross-comparison) for this object. This will also set the
    #     translation kernel that enables broadcasting the images over the
    #     displacements. The resulting grid will be square in shape, with a
    #     maximum displacement in both X and Y of max_displacement_pixels,
    #     but the number of displacements in either direction will vary. The
    #     original image (a displacement of (0, 0)) will always be included.

    #     Args:
    #         max_displacement_pixels (float): Maximum displacement value
    #         n_displacements_x (int): Number of displacements in the x-direction
    #         n_displacements_y (int): Number of displacements in the y-direction
    #         pixel_size_angstrom (float | None, optional): Size of each pixel, in
    #             angstrom. Needed to interpret the max_displacement_pixels. Only
    #             required if this image stack does not have a physical grid set.
    #     """
    #     if pixel_size_angstrom is None:
    #         if getattr(self, 'phys_grid', None) is None:
    #             raise ValueError("If pixel size is not provided, a physical grid must be set.")
    #         # NOTE: Assumes square pixels!
    #         pixel_size_angstrom = self.phys_grid.pixel_size[0]
    #     assert pixel_size_angstrom is not None
    #     if self.box_size[0] != self.box_size[1]:
    #         print("Displacement grid currently only implemented for square viewing box. Results may be unreliable.")
    #         # raise NotImplementedError("Displacement grid currently only implemented for square viewing box.")

    #     max_d_angstrom = max_displacement_pixels * pixel_size_angstrom
    #     n_displacements_x = max(1, n_displacements_x)
    #     n_displacements_y = max(1, n_displacements_y)
    #     self.n_displacements = n_displacements_x * n_displacements_y
    #     if self.n_displacements == 1:
    #         max_d_angstrom = 0.

    #     x_disp = max_d_angstrom if n_displacements_x > 1 else 0.
    #     y_disp = max_d_angstrom if n_displacements_y > 1 else 0.
    #     _x = torch.linspace( -x_disp, x_disp, n_displacements_x)
    #     _y = torch.linspace( -y_disp, y_disp, n_displacements_y)
    #     _X, _Y = torch.meshgrid(_x, _y, indexing="xy")
    #     _X = _X.flatten()
    #     _Y = _Y.flatten()
    #     assert self.n_displacements == _X.size()[0]

    #     if n_displacements_x % 2 == 0 or n_displacements_y % 2 == 0:
    #         # even number of steps means we omitted (0, 0); add it back in
    #         _X = torch.cat((_X, torch.tensor([0.])))
    #         _Y = torch.cat((_Y, torch.tensor([0.])))
    #         self.n_displacements += 1
    #     self.displacement_grid_angstrom = torch.stack([_X, _Y])
    #     if getattr(self, 'polar_grid', None) is None:
    #         print("Warning: No polar grid set. Translation matrix is invalid.")
    #         return
    #     self._translation_matrix = self.polar_grid.get_fourier_translation_kernel(
    #         _X,
    #         _Y,
    #         self.box_size[0],
    #         self.box_size[1],
    #         Precision.SINGLE if self.images_fourier.dtype == torch.complex64 else Precision.DOUBLE,
    #         self.images_fourier.device
    #     )


    # def project_images_over_displacements(
    #     self,
    #     range_min: int,
    #     range_max_excl: int,
    #     device: torch.device | str | None = None
    # ):
    #     """Returns a tensor of the requested range of the Fourier-space images,
    #     projected over the currently configured displacements, stored on the
    #     requested device.

    #     Args:
    #         range_min (int): Index of first image to return
    #         range_max_excl (int): One more than the index of the last image
    #             to be returned
    #         device (torch.device | str | None): Device on which to store the
    #             returned tensor. If unset, will default to the device where
    #             the current Fourier images reside.

    #     Returns:
    #         torch.Tensor: A tensor of Fourier-space images, projected over
    #             the configured displacements. This tensor will be indexed
    #             as [image, displacement, radius, inplane-value], with the
    #             latter two dimensions corresponding to the polar quadrature
    #             grid.
    #     """

    #     if getattr(self, '_translation_matrix', None) is None:
    #         raise ValueError("Translation kernel was never set. Most likely this image object lacks a polar grid.")
    #     if device is None:
    #         device = self.images_fourier.device
    #     return (self.images_fourier[range_min:range_max_excl].unsqueeze(1) * self._translation_matrix).to(device)


    # def _verify_displacements(
    #     self,
    #     x_displacements: FloatArrayType | torch.Tensor | float,
    #     y_displacements: FloatArrayType | torch.Tensor | float,
    #     precision: Precision,
    #     device: torch.device,
    #     displacement_per_image: bool = False
    # ):
    #     x_disp = to_torch(x_displacements, precision, device)
    #     y_disp = to_torch(y_displacements, precision, device)

    #     if displacement_per_image:
    #         if x_disp.shape[0] != self.n_images or y_disp.shape[0] != self.n_images:
    #             raise ValueError("Per-image displacements must provide one displacement per image.")
    #         return (x_disp, y_disp)
    #     return (x_disp.sum().unsqueeze(0), y_disp.sum().unsqueeze(0))


    # def displace_fourier_images(
    #     self,
    #     x_displacements: FloatArrayType | torch.Tensor | float,
    #     y_displacements: FloatArrayType | torch.Tensor | float,
    #     displacement_per_image: bool = False
    # ):
    #     """Apply a displacement to the Fourier-space images in the collection, modifying them in-place.
    #     The displacements are assumed to be intended as a fraction of half a viewing box (i.e. they
    #     will be multiplied by 2 and divided by the viewing box size)

    #     Args:
    #         x_displacements (FloatArrayType | torch.Tensor | float): Displacements to apply. May be constant
    #             for each pixel, or a set of displacements to apply per-pixel for each image, or a complete
    #             tensor of displacements.
    #         y_displacements (FloatArrayType | torch.Tensor | float): Displacements to apply. May be constant
    #             for each pixel, or a set of displacements to apply per-pixel for each image, or a complete
    #             tensor of displacements.
    #         displacement_per_image (bool, optional): If True, enforce that the displacements are equal-length
    #             tensors with one element each per image. Otherwise, assume they are a set of displacements to
    #             sum and apply to each image.
    #     """
    #     self._ensure_fourier_images()
    #     precision = Precision.SINGLE if self.images_fourier.dtype == torch.complex64 else Precision.DOUBLE
    #     device = self.images_fourier.device

    #     (x_disp, y_disp) = self._verify_displacements(x_displacements, y_displacements, precision, device, displacement_per_image)
    #     translation_kernel = self.polar_grid.get_fourier_translation_kernel(
    #         x_disp,
    #         y_disp,
    #         self.box_size[0],
    #         self.box_size[1],
    #         precision,
    #         device
    #     )
    #     self.images_fourier = (self.images_fourier * translation_kernel)




    # def normalize_images_fourier(
    #     self,
    #     ord: int = 1,
    #     use_max: bool = False,
    # ):
    #     """Normalize the Fourier-space images in the collection.

    #     Args:
    #         ord (int, optional): Degree of norm to apply. Defaults to 1.
    #         use_max (bool, optional): Whether to use the max in place of an LP norm. Defaults to False.

    #     Returns:
    #         torch.Tensor: The norm applied to the images (which are modified in-place).
    #     """
    #     self._ensure_fourier_images()
    #     if use_max:
    #         maxval = get_imgs_max(self.images_fourier)
    #         self.images_fourier /= maxval
    #         return maxval
    #     else:
    #         lpnorm = self.polar_grid.integrate(self.images_fourier.abs().pow(ord)).pow(1.0 / ord)
    #         for i in range(len(self.images_fourier.shape) - 1):
    #             # lpnorm = np.expand_dims(lpnorm, axis = -1)
    #             lpnorm = torch.unsqueeze(lpnorm, dim = -1)
    #         self.images_fourier /= lpnorm
    #         return lpnorm


    # def _make_rotation_tensor(self, inplane_rotations: np.ndarray | torch.Tensor | float) -> torch.Tensor:
    #     if np.isscalar(inplane_rotations):
    #         _rotations = torch.tensor([inplane_rotations], dtype = torch.double)
    #     elif isinstance(inplane_rotations, np.ndarray):
    #         _rotations = torch.from_numpy(inplane_rotations).to(dtype = torch.double)
    #     else:
    #         assert isinstance(inplane_rotations, torch.Tensor)
    #         _rotations = inplane_rotations.to(dtype=torch.double)
    #     if _rotations.ndim > 1:
    #         raise ValueError("inplane_rotations must be a 1D array.")
    #     if _rotations.shape[0] != self.images_fourier.shape[0]:
    #         if _rotations.shape[0] == 1:
    #             # Manually broadcast rotations
    #             _rotations = _rotations * torch.ones(self.images_fourier.shape[0])
    #         else:
    #             raise ValueError("Number of rotations must be equal to the number of images.")
    #     _rotations.to(self.images_fourier.device)

    #     return _rotations


    # def rotate_images_fourier_discrete(
    #     self,
    #     inplane_rotations: np.ndarray | torch.Tensor | float
    # ):
    #     """Apply an in-plane rotation to the Fourier-space images.

    #     While a continuous value is accepted, the resulting rotated object must align
    #     with the points on the polar grid, so the rotation will be discretized.

    #     Assumes a uniform polar grid.

    #     Args:
    #         inplane_rotations (np.ndarray | torch.Tensor | float): Rotation to apply,
    #             in revolutions. If an array or tensor, must be of length 1 (in which
    #             case it will be applied to all images) or a 1-d array of length equal
    #             to the number of images (one rotation per image).

    #     Raises:
    #         ValueError: If the inplane_rotations is multi-dimensional, or cannot be
    #             obviously broadcast to the number of images.
    #     """
    #     self._ensure_fourier_images()
    #     _rotations = self._make_rotation_tensor(inplane_rotations)

    #     inplane_rotation_step = 2 * np.pi / self.polar_grid.n_inplanes
    #     inplane_rotations_discrete = -torch.round(_rotations / inplane_rotation_step)
    #     for i in range(self.n_images):
    #         self.images_fourier[i] = torch.roll(self.images_fourier[i], int(inplane_rotations_discrete[i]), dims = 1)

    
    # ## This function is used for specific datasets, need more documentation to expose it to users
    # def filter_padded_images(self, rtol = 1e-1):
    #     """Restrict the image collection to the set of images which do not have padding.
    #     """
    #     ## this is to remove the artifacted images in some dataset on EMPIAR
    #     ## find padded physical images and remove them
    #     if not self.has_physical_images():
    #         return
    #     not_padded = np.ones(self.images_phys.shape[0], dtype=bool)
    #     for i in range(self.images_phys.shape[0]):
    #         if (torch.allclose(self.images_phys[i, 0, :], self.images_phys[i, 1, :], rtol = rtol) or
    #             torch.allclose(self.images_phys[i, :, 0], self.images_phys[i, :, 1], rtol = rtol) or
    #             torch.allclose(self.images_phys[i,-1, :], self.images_phys[i,-2, :], rtol = rtol) or
    #             torch.allclose(self.images_phys[i, :,-1], self.images_phys[i, :,-2], rtol = rtol)
    #         ):
    #             not_padded[i] = False
    #     # print(f"Number of not padded images: {np.sum(not_padded)}")
    #     self.images_phys = self.images_phys[not_padded]
    #     self.n_images = self.images_phys.shape[0]
    #     return not_padded


    # def get_power_spectrum(self):
    #     """Gets the power spectrum of the (Fourier-space) images.

    #     Raises:
    #         ValueError: If no Fourier-space images exist in the collection.

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: Tuple of power-spectrum values and the resolutions.
    #     """
    #     if self.images_fourier.shape[0] == 0:
    #         raise ValueError("Fourier images not found. Please transform the images to Fourier domain before calculating the power spectrum.")
    #     resolutions = np.amax(self.box_size) / (2.0 * self.polar_grid.radius_shells)
    #     images_fourier = self.images_fourier
    #     power_spectrum: ComplexArrayType = torch.mean(torch.abs(images_fourier) ** 2, dim = (0, 2)).cpu().numpy()
    #     return power_spectrum, resolutions
    

    