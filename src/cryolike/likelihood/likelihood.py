import numpy as np
from scipy.special import loggamma as lgamma
import torch
from typing import Optional

from .cross_correlation import calc_cross_correlation
from cryolike.grid import UniformPolarGrid
from cryolike.pose import Displacements2D
from cryolike.util import PrecisionLevel, get_float_dtype, get_complex_dtype, absq, complex_mul_real


def _identity_kernel(polar_grid: UniformPolarGrid):
    return 4.0 * torch.sinc(2.0 * polar_grid.x_points) * torch.sinc(2.0 * polar_grid.y_points)


def likelihood_all_poses(
    polar_grid: UniformPolarGrid,
    box_size: float,
    n_pixels_phys: int,
    images_fourier: torch.Tensor, # (n_images, n_shells, n_inplanes)
    templates_fourier: torch.Tensor, # (n_templates, n_shells, n_inplanes)
    ctf_tensor: torch.Tensor, # (n_params, n_shells, n_inplanes) ## assume anisotropic CTF, slower
    displacements: Optional[Displacements2D] = None,
    precision: PrecisionLevel = PrecisionLevel.SINGLE,
    device: Optional[str | torch.device] = None
):
    """Compute the per-image integrated log likelihood.
    """

    device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    float_type = get_float_dtype(precision)
    complex_type = get_complex_dtype(precision)
    if displacements is None:
        displacements = Displacements2D(
            torch.zeros((1), dtype=float_type, device=device),
            torch.zeros((1), dtype=float_type, device=device),
            box_size_angstrom = box_size
        )
    
    assert (images_fourier.shape[0] == ctf_tensor.shape[0]) or ctf_tensor.shape[0] == 1
    assert images_fourier.shape[1] == polar_grid.n_shells
    assert images_fourier.shape[2] == polar_grid.n_inplanes
    assert templates_fourier.shape[1] == polar_grid.n_shells
    assert templates_fourier.shape[2] == polar_grid.n_inplanes
    assert ctf_tensor.shape[1] == polar_grid.n_shells
    assert ctf_tensor.shape[2] == polar_grid.n_inplanes
    assert images_fourier.dtype == complex_type
    assert templates_fourier.dtype == complex_type
    assert ctf_tensor.dtype in [float_type, complex_type]

    polar_grid = polar_grid.to(dtype=float_type, device=device)
    weights = polar_grid.weight_shells / polar_grid.n_inplanes
    displacement_kernels = displacements.kernel(polar_grid)
    s_points = _identity_kernel(polar_grid)
    Iss = torch.sum(absq(s_points) * weights[:,None])

    ctf_templates_fourier = ctf_tensor.unsqueeze(1) * templates_fourier.unsqueeze(0)
    Ixx = torch.sum(
        absq(ctf_templates_fourier) * weights[None,None,:,None],
        dim=(-2, -1)
    )
    Iyy = torch.sum(
        absq(images_fourier) * weights[None,:,None],
        dim=(-2, -1)
    )

    displaced_templates_fourier = templates_fourier.unsqueeze(1) * displacement_kernels.unsqueeze(0)
    displaced_templates_bessel = torch.fft.fft(displaced_templates_fourier, dim=-1, norm="ortho")
    ctf_images_fourier = ctf_tensor * images_fourier
    ctf_images_bessel_conj = torch.fft.fft(ctf_images_fourier, dim=-1, norm="ortho").conj()

    Ixy = calc_cross_correlation(
        ctf_images_bessel_conj=ctf_images_bessel_conj,
        displaced_templates_bessel=displaced_templates_bessel,
        weights=weights,
        n_inplanes=polar_grid.n_inplanes
    ) # (n_images, n_templates, n_displacements, n_inplanes)

    s_points_weights = s_points[None,:,:] * weights[None,:,None]
    Isx = torch.sum(s_points_weights.unsqueeze(0) * ctf_templates_fourier, dim=(-2, -1))
    Isy = torch.sum(s_points_weights * images_fourier, dim=(-2, -1))

    # ## Unify to (n_images, n_templates, n_displacements, n_inplanes)
    Ixx = Ixx[:,:,None,None]
    Iyy = Iyy[:,None,None,None]
    Isx = Isx[:,:,None,None]
    Isy = Isy[:,None,None,None]

    print("Ixx: ", Ixx.amin(), Ixx.amax())
    print("Iyy: ", Iyy.amin(), Iyy.amax())
    print("Ixy: ", Ixy.amin(), Ixy.amax())
    print("Isx.real: ", Isx.real.amin(), Isx.real.amax(), "Isx.imag: ", Isx.imag.amin(), Isx.imag.amax())
    print("Isy.real: ", Isy.real.amin(), Isy.real.amax(), "Isy.imag: ", Isy.imag.amin(), Isy.imag.amax())
    print("Iss: ", Iss.amin(), Iss.amax())

    A = - absq(Isx) + Ixx * Iss
    B = - complex_mul_real(Isx, Isy) + Ixy * Iss
    C =   absq(Isy) - Iyy * Iss
    D = - (B ** 2 / A + C)
    print("A: ", A.amin(), A.amax())
    print("B^2: ", (B**2).amin(), (B**2).amax())
    print("C: ", C.amin(), C.amax())
    print("D: ", D.amin(), D.amax())
    
    p = n_pixels_phys / 2.0 - 2.0
    constant = (3.0 - n_pixels_phys) / 2.0 * np.log(2 * np.pi) \
                - np.log(2) - 0.5 * np.log(Iss) \
                + lgamma(n_pixels_phys / 2.0 - 2.0) \
                + p * np.log(2 * Iss)
    log_likelihood_msdw = -p * torch.log(D) - 0.5 * torch.log(A) + constant
    
    return log_likelihood_msdw


# def likelihood_single(
#     polar_grid: UniformPolarGrid,
#     n_pixels_phys: int,
#     images_fourier: torch.Tensor, # (n_images, n_shells, n_inplanes)
#     templates_fourier: torch.Tensor, # (n_images, n_shells, n_inplanes), generated ahead
#     precision: PrecisionLevel = PrecisionLevel.SINGLE,
#     device: Optional[str | torch.device] = None
# ):
#     """Compute the per-image integrated log likelihood.
#     """

#     device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     float_type = get_float_dtype(precision)
#     complex_type = get_complex_dtype(precision)

#     assert images_fourier.shape[0] == templates_fourier.shape[0]
#     assert images_fourier.shape[1] == polar_grid.n_shells
#     assert images_fourier.shape[2] == polar_grid.n_inplanes
#     assert templates_fourier.shape[1] == polar_grid.n_shells
#     assert templates_fourier.shape[2] == polar_grid.n_inplanes
#     assert images_fourier.dtype == complex_type
#     assert templates_fourier.dtype == complex_type

#     polar_grid = polar_grid.to(dtype=float_type, device=device)
#     weights = polar_grid.weight_shells[None,:,None] / polar_grid.n_inplanes
    
#     s_points = 4.0 * torch.sinc(2.0 * polar_grid.x_points) * torch.sinc(2.0 * polar_grid.y_points)
#     Iss = torch.sum(absq(s_points) * weights)
#     s_points = s_points.unsqueeze(0)

#     Ixx = torch.sqrt(torch.sum(
#         absq(templates_fourier) * weights,
#         dim=(-2, -1)
#     )) # |template|^2
#     Iyy = torch.sqrt(torch.sum(
#         absq(images_fourier) * weights,
#         dim=(-2, -1)
#     )) # |image|^2
#     templates_bessel = torch.fft.fft(templates_fourier, dim=-1, norm="ortho")
#     images_bessel = torch.fft.fft(images_fourier, dim=-1, norm="ortho")
#     Ixy = torch.sum((
#         images_bessel.conj() * templates_bessel * weights
#     ), dim=(-2, -1))
#     Isx = torch.sum(s_points * templates_fourier * weights, dim=(-2, -1))
#     Isy = torch.sum(s_points * images_fourier * weights, dim=(-2, -1))
#     A = - absq(Isx) + Ixx * Iss
#     B = - complex_mul_real(Isx, Isy) + Ixy * Iss
#     C =   absq(Isy) - Iyy * Iss
#     D = - (B ** 2 / A + C)
#     p = n_pixels_phys / 2.0 - 2.0
#     constant = (3.0 - n_pixels_phys) / 2.0 * np.log(2 * np.pi) \
#                 - np.log(2) - 0.5 * np.log(Iss) \
#                 + lgamma(n_pixels_phys / 2.0 - 2.0) \
#                 + p * np.log(2 * Iss)
#     log_likelihood_s = -p * torch.log(D) - 0.5 * torch.log(A) + constant
#     return log_likelihood_s