import torch
from typing import Optional

from cryolike.grid import UniformPolarGrid
from cryolike.pose import Displacements2D
from cryolike.util import PrecisionLevel, get_float_dtype, get_complex_dtype, absq, complex_mul_real

def calc_cross_correlation(
    ctf_images_bessel_conj: torch.Tensor,
    displaced_templates_bessel: torch.Tensor,
    weights: torch.Tensor,
    n_inplanes: int
):
    ## Compute cross-correlation between image and template
    cross_correlation_bessel = torch.einsum(
        "mnq,sdnq,n->msdq",
        ctf_images_bessel_conj, # (m,n,q) = (n_images, n_shells, n_inplanes)
        displaced_templates_bessel, # (s,d,n,q) = (n_templates, n_displacements, n_shells, n_inplanes)
        weights + 0j
    ) # (m,s,d,q) = (n_images, n_templates, n_displacements, n_inplanes)
    cross_correlation_fourier = torch.fft.irfft(
        cross_correlation_bessel,
        n=n_inplanes,
        dim=-1,
        norm="forward"
    ) # (m,s,d,w) = (n_images, n_templates, n_displacements, n_inplanes)
    return cross_correlation_fourier


def cross_correlation_images_templates(
    polar_grid: UniformPolarGrid,
    box_size: float,
    images_fourier: torch.Tensor, # (n_images, n_shells, n_inplanes)
    templates_fourier: torch.Tensor, # (n_templates, n_shells, n_inplanes)
    ctf_tensor: torch.Tensor, # (n_params, n_shells, n_inplanes) ## assume anisotropic CTF, slower
    displacements: Optional[Displacements2D] = None,
    precision: PrecisionLevel = PrecisionLevel.SINGLE,
    device: Optional[str | torch.device] = None
) -> torch.Tensor:
    
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

    ctf_templates_fourier = ctf_tensor.unsqueeze(1) * templates_fourier.unsqueeze(0)
    templates_norm = torch.sqrt(torch.sum(
        absq(ctf_templates_fourier) * weights[None,None,:,None],
        dim=(-2, -1)
    ))
    images_norm = torch.sqrt(torch.sum(
        absq(images_fourier) * weights[None,:,None],
        dim=(-2, -1)
    ))

    displaced_templates_fourier = templates_fourier.unsqueeze(1) * displacement_kernels.unsqueeze(0)
    displaced_templates_bessel = torch.fft.fft(displaced_templates_fourier, dim=-1, norm="ortho")
    ctf_images_fourier = ctf_tensor * images_fourier
    ctf_images_bessel_conj = torch.fft.fft(ctf_images_fourier, dim=-1, norm="ortho").conj()

    cross_correlation = calc_cross_correlation(
        ctf_images_bessel_conj=ctf_images_bessel_conj,
        displaced_templates_bessel=displaced_templates_bessel,
        weights=weights,
        n_inplanes=polar_grid.n_inplanes
    )
    cross_correlation_fourier_normalized = cross_correlation / images_norm[:,None,None,None] / templates_norm[:,:,None,None]
    return cross_correlation_fourier_normalized

