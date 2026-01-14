import torch
from numpy import pi
import pytest

from cryolike.pose import Displacements2D
from cryolike.likelihood import cross_correlation_images_templates
from cryolike.util import get_device, get_float_dtype, get_complex_dtype

from cross_correlation_fixtures import (
    parameters,
    make_cases,
    make_polar_grid,
    make_viewing_angles,
    make_planewave_templates,
    viewing_angles_to_cartesian_displacements,
    get_planar_ctf,
    planewave_planar_planewave_planar
)


param_matrix = make_cases()
@pytest.mark.parametrize("params", param_matrix)
def test_cross_correlation_PxxP_from_a_k_p(params: parameters):

    device = get_device(params.device)
    box_size = 2.0
    n_displacements_per_axis = 3
    
    wavevector_planewave = params.wavevector.to(dtype=params.float_type, device=device)
    angle_planar_ctf_template = - pi / 5.0 # phi_S
    angle_planar_ctf_image = + pi / 3.0  # phi_M
    displacement_planewave_image = torch.tensor([-0.17, -0.03], dtype=params.float_type, device=device) # delta_M
    
    polar_grid = make_polar_grid(params.n_pixels).to(dtype=params.float_type, device=device)
    viewing_angles = make_viewing_angles(device, params.float_type)
    planar_ctf_template = get_planar_ctf(polar_grid, angle_planar_ctf_template).unsqueeze(0)
    planar_ctf_image = get_planar_ctf(polar_grid, angle_planar_ctf_image).unsqueeze(0)
    templates_fourier = make_planewave_templates(wavevector_planewave, viewing_angles, polar_grid)

    displacements_image = Displacements2D(
        x_displacements_angstrom=displacement_planewave_image[0].unsqueeze(0),
        y_displacements_angstrom=displacement_planewave_image[1].unsqueeze(0),
        box_size_angstrom=box_size
    ).to(dtype=params.float_type, device=device)
    displacement_kernel_image = displacements_image.kernel(polar_grid)
    images_fourier = templates_fourier.clone() * displacement_kernel_image * planar_ctf_image
    
    displacements_templates = Displacements2D.sample_grid(
        max_displacements_angstrom = params.max_displacement,
        n_samples_per_axis = n_displacements_per_axis,
        box_size_angstrom = box_size
    ).to(dtype=params.float_type, device=device)

    wavevector_planewave_templates = viewing_angles_to_cartesian_displacements(viewing_angles, wavevector_planewave).to(device)
    wavevector_planewave_images = wavevector_planewave_templates.clone() - displacement_planewave_image

    analytic = planewave_planar_planewave_planar(
        wavevector_planewave_templates,
        wavevector_planewave_images,
        polar_grid.theta_shell * -1.0,
        displacements_templates.xy_displacements_angstrom,
        torch.tensor(angle_planar_ctf_template, dtype=params.float_type, device=params.device),
        torch.tensor(angle_planar_ctf_image, dtype=params.float_type, device=params.device),
        polar_grid.radius_max
    )
    cc = cross_correlation_images_templates(
        polar_grid=polar_grid,
        box_size=box_size,
        images_fourier=images_fourier,
        templates_fourier=templates_fourier,
        ctf_tensor=planar_ctf_template,
        displacements=displacements_templates,
        precision=params.precision,
        device=device
    )
    assert torch.allclose(cc, analytic, atol=params.abs_tolerance_cross_correlation, rtol=params.rel_tolerance_cross_correlation)


def test_cross_correlation_gradient():

    params = parameters.default()
    device = get_device(params.device)
    box_size = 2.0
    n_displacements_per_axis = 1
    
    wavevector_planewave = params.wavevector.to(dtype=params.float_type, device=device)
    angle_planar_ctf_template = - pi / 5.0 # phi_S
    angle_planar_ctf_image = + pi / 3.0  # phi_M
    displacement_planewave_image = torch.tensor([-0.17, -0.03], dtype=params.float_type, device=device) # delta_M
    
    polar_grid = make_polar_grid(params.n_pixels).to(dtype=params.float_type, device=device)
    viewing_angles = make_viewing_angles(device, params.float_type)
    planar_ctf_template = get_planar_ctf(polar_grid, angle_planar_ctf_template).unsqueeze(0)
    planar_ctf_image = get_planar_ctf(polar_grid, angle_planar_ctf_image).unsqueeze(0)
    templates_fourier = make_planewave_templates(wavevector_planewave, viewing_angles, polar_grid).requires_grad_(True)

    displacements_image = Displacements2D(
        x_displacements_angstrom=displacement_planewave_image[0].unsqueeze(0),
        y_displacements_angstrom=displacement_planewave_image[1].unsqueeze(0),
        box_size_angstrom=box_size
    ).to(dtype=params.float_type, device=device)
    displacement_kernel_image = displacements_image.kernel(polar_grid)
    images_fourier = templates_fourier.clone() * displacement_kernel_image * planar_ctf_image
    
    displacements_templates = Displacements2D.sample_grid(
        max_displacements_angstrom = params.max_displacement,
        n_samples_per_axis = n_displacements_per_axis,
        box_size_angstrom = box_size
    ).to(dtype=params.float_type, device=device)

    ## test gradient
    cc = cross_correlation_images_templates(
        polar_grid=polar_grid,
        box_size=box_size,
        images_fourier=images_fourier,
        templates_fourier=templates_fourier,
        ctf_tensor=planar_ctf_template,
        displacements=displacements_templates,
        precision=params.precision,
        device=device
    )
    log_lik = torch.log(1 - cc ** 2).sum().backward()
    assert templates_fourier.grad is not None
    
