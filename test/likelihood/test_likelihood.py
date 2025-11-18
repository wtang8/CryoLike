import numpy as np
import torch
import pytest
from unittest.mock import patch
from numpy import pi

from cryolike.grid import UniformPolarGrid
from cryolike.pose import Displacements2D
from cryolike.likelihood.likelihood import likelihood_all_poses, _identity_kernel
from cryolike.util import get_device

from cross_correlation_fixtures import (
    parameters,
    make_cases,
    make_polar_grid,
    make_viewing_angles,
    make_planewave_templates,
    viewing_angles_to_cartesian_displacements,
    get_planar_ctf,
)
from likelihood_fixtures import LogLikelihoodPlanarCTFPlanewaves


wavevector_planewave_identity = torch.tensor([0.01, 0.01]) * (2.0 * np.pi)
def custom_identity_kernel(polar_grid: UniformPolarGrid):
    _wv = wavevector_planewave_identity.to(dtype=polar_grid.x_points.dtype, device=polar_grid.x_points.device)
    return torch.exp(
        2 * np.pi * 1j * (
            polar_grid.x_points * _wv[0] + 
            polar_grid.y_points * _wv[1])
    )
    

param_matrix = make_cases()
@pytest.mark.parametrize("params", param_matrix)
@patch('cryolike.likelihood.likelihood._identity_kernel', custom_identity_kernel)
def test_likelihood_PxxP_from_a_k_p(params: parameters):

    device = get_device(params.device)
    box_size = 2.0
    n_displacements_per_axis = 3
    
    wavevector_planewave = params.wavevector.to(dtype=params.float_type, device=device)
    angle_planar_ctf_template = - pi / 5.0 # phi_S
    angle_planar_ctf_image = + pi / 3.0  # phi_M
    n_pixels_total = params.n_pixels * params.n_pixels
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
    
    _gamma = polar_grid.theta_shell * -1.0
    displacements = displacements_templates.xy_displacements_angstrom
    
    log_likelihood_class = LogLikelihoodPlanarCTFPlanewaves(
        wavevector_planewave_templates=wavevector_planewave_templates,
        wavevector_planewave_images=wavevector_planewave_images,
        wavevector_planewave_identity=wavevector_planewave_identity.unsqueeze(0),
        angle_planar_ctf_template=angle_planar_ctf_template,
        angle_planar_ctf_image=angle_planar_ctf_image,
        gamma=_gamma,
        displacements=displacements,
        polar_grid=polar_grid,
        n_pixels=n_pixels_total,
        precision=params.precision,
    )
    log_likelihood_analytical = log_likelihood_class.log_likelihood_final()

    log_likelihood_msdw = likelihood_all_poses(
        polar_grid=polar_grid,
        box_size=box_size,
        n_pixels_phys=n_pixels_total,
        images_fourier=images_fourier,
        templates_fourier=templates_fourier,
        ctf_tensor=planar_ctf_template,
        displacements=displacements_templates,
        precision=params.precision,
        device=device
    )
    assert torch.allclose(log_likelihood_msdw, log_likelihood_analytical, atol=params.abs_tolerance_log_likelihood, rtol=params.rel_tolerance_log_likelihood)


# if __name__ == '__main__':
#     print('running test_cross_correlation_PxxP_from_a_k_p')
#     # params = parameters.default()
#     # test_likelihood_PxxP_from_a_k_p(params)
#     # print('returning')
#     # pytest.main([__file__, "-v", "--tb=short"])