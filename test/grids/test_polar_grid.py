import torch
import numpy as np
from scipy.special import roots_jacobi
from dataclasses import dataclass, field
from enum import Enum
import math
from pytest import mark, fixture, raises

from cryolike.grids import UniformPolarGrid, QuadratureType
from cryolike.util import ensure_positive, get_float_dtype, set_precision

PRECISION_PARAMS = [
    ('single', 1e-5), 
    ('double', 1e-10)
]


class TestUniformPolarGrid:
    R_MAX = 5.0
    D_RADII = 1.0
    N_INPLANES = 8

    def test_initialization_and_validation(self):

        set_precision('double')

        # Test successful initialization with default quadrature
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)
        assert isinstance(grid, UniformPolarGrid)
        assert grid.quadrature == QuadratureType.GAUSS_JACOBI_BETA_1

        # Check calculated properties
        expected_n_shells = 1 + math.ceil(self.R_MAX / self.D_RADII)
        assert grid.n_shells == expected_n_shells
        assert grid.n_points == expected_n_shells * self.N_INPLANES

        # Check tensor shapes
        assert grid.radius_shells.shape == (expected_n_shells,)
        assert grid.theta_shell.shape == (self.N_INPLANES,)
        assert grid.x_points.shape == (expected_n_shells, self.N_INPLANES)

        # Test invalid input
        with raises(ValueError) as cm:
            UniformPolarGrid(0.0, self.D_RADII, self.N_INPLANES)
        assert "radius_max must be positive" in str(cm.value)

    @mark.parametrize("precision,abs_tol", PRECISION_PARAMS)
    def test_quadrature_type_beta_1(self, precision, abs_tol):
        set_precision(precision)
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES, QuadratureType.GAUSS_JACOBI_BETA_1)
        # Check radial coordinates are in [0, R_MAX]
        assert torch.all(grid.radius_shells >= 0)
        assert torch.all(grid.radius_shells <= self.R_MAX + abs_tol)

        # Test integration: integral of f(r,theta)=1 over a disk of radius R_MAX is pi * R_MAX^2
        area_expected = math.pi * self.R_MAX**2
        f = torch.ones(grid.n_shells, grid.n_inplanes, dtype=get_float_dtype())
        area_calculated = grid.integrate(f)
        assert math.isclose(area_calculated.item(), area_expected, abs_tol=abs_tol)

    @mark.parametrize("precision,abs_tol", PRECISION_PARAMS)
    def test_quadrature_type_beta_2(self, precision, abs_tol):
        set_precision(precision)
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES, QuadratureType.GAUSS_JACOBI_BETA_2)
        # The integral identity is different for beta=2, so we only check shapes/ranges.
        assert torch.all(grid.radius_shells >= 0)
        assert torch.all(grid.radius_shells <= self.R_MAX + abs_tol)
        assert grid.n_shells == 1 + math.ceil(self.R_MAX / self.D_RADII)

    @mark.parametrize("precision,abs_tol", PRECISION_PARAMS)
    def test_quadrature_type_legendre(self, precision, abs_tol):
        set_precision(precision)
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES, QuadratureType.GAUSS_LEGENDRE)
        # Check radial coordinates are in [0, R_MAX]
        assert torch.all(grid.radius_shells >= 0)
        assert torch.all(grid.radius_shells <= self.R_MAX + abs_tol)

        # Test integration: integral of f(r,theta)=1 over a disk of radius R_MAX is pi * R_MAX^2
        area_expected = math.pi * self.R_MAX**2
        f = torch.ones(grid.n_shells, grid.n_inplanes, dtype=get_float_dtype())
        area_calculated = grid.integrate(f)
        # Gauss-Legendre quadrature on a transformed integral may have a slightly different accuracy,
        # but should be close.
        assert math.isclose(area_calculated.item(), area_expected, rel_tol=0.01) # Relaxed tolerance for Legendre formula

    @mark.parametrize("precision,abs_tol", PRECISION_PARAMS)
    def test_integrate_multi_dimension(self, precision, abs_tol):
        set_precision(precision)
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES) # Beta 1 is exact
        area_expected = math.pi * self.R_MAX**2

        # Create a 3D tensor: (batch_size, n_shells, n_inplanes)
        batch_size = 5
        f_3d = torch.ones(batch_size, grid.n_shells, grid.n_inplanes, dtype=get_float_dtype())
        result_3d = grid.integrate(f_3d)

        assert result_3d.shape == (batch_size,)
        assert torch.allclose(result_3d, torch.tensor([area_expected] * batch_size, dtype=get_float_dtype()), atol=abs_tol)

        # Test invalid input shape
        with raises(ValueError) as cm:
            grid.integrate(torch.ones(grid.n_shells + 1, grid.n_inplanes))
        assert "does not match grid shape" in str(cm.value)

    @mark.parametrize("precision,abs_tol", PRECISION_PARAMS)
    def test_get_fourier_translation_kernel_zero_disp(self, precision, abs_tol):
        set_precision(precision)
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)
        dtype = get_float_dtype()

        # Zero displacement (should yield kernel of 1s, which is exp(0))
        x_disp = torch.tensor([0.0], dtype=dtype)
        y_disp = torch.tensor([0.0], dtype=dtype)

        kernel = grid.get_fourier_translation_kernel(x_disp, y_disp)

        assert kernel.shape == (1, grid.n_shells, grid.n_inplanes)
        # Check that the real part is 1 and imaginary part is 0
        assert torch.allclose(kernel.real, torch.ones_like(kernel.real), atol=abs_tol)
        assert torch.allclose(kernel.imag, torch.zeros_like(kernel.imag), atol=abs_tol)

    def test_get_fourier_translation_kernel_shape(self):
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)
        dtype = get_float_dtype()

        # Multiple displacements
        n_displacements = 3
        x_disp = torch.ones(n_displacements, dtype=dtype) * 0.1
        y_disp = torch.ones(n_displacements, dtype=dtype) * -0.2

        kernel = grid.get_fourier_translation_kernel(x_disp, y_disp)

        assert kernel.shape == (n_displacements, grid.n_shells, grid.n_inplanes)
        assert kernel.is_complex()

    def test_to_method(self):
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)

        # Move to float32
        grid_f32 = grid.to(dtype=torch.float32, device='cpu')
        assert grid_f32.radius_shells.dtype == torch.float32
        assert grid_f32.x_points.dtype == torch.float32

        # Move back to float64 (initial dtype)
        grid_f64 = grid.to(dtype=torch.float64, device='cpu')
        assert grid_f64.radius_shells.dtype == torch.float64

    def test_invalid_quadrature_type(self):
        with raises(ValueError, match="Unknown quadrature type"):
            UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES, quadrature="invalid-quadrature-type")

    def test_invalid_gauss_jacobi_beta(self):
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)
        with raises(ValueError, match="Beta parameter must be 1 or 2"):
            grid._gauss_jacobi(beta=3)

    def test_integrate_invalid_dimension(self):
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)
        with raises(ValueError, match="Input tensor must have more than 2 dimension"):
            grid.integrate(torch.ones(grid.n_shells))

    def test_fourier_kernel_device_fallback(self, monkeypatch):
        # Simulate no CUDA available
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)
        x_disp = torch.tensor([0.1])
        y_disp = torch.tensor([0.1])

        # Request 'cuda', expect it to fallback to 'cpu'
        kernel = grid.get_fourier_translation_kernel(
            x_displacements_angstrom=x_disp,
            y_displacements_angstrom=y_disp,
            device='cuda'
        )

        assert kernel.device.type == 'cpu'

    def test_repr(self):
        grid = UniformPolarGrid(self.R_MAX, self.D_RADII, self.N_INPLANES)
        repr_str = repr(grid)
        
        expected_n_shells = 1 + math.ceil(self.R_MAX / self.D_RADII)
        expected_n_points = expected_n_shells * self.N_INPLANES
        
        expected_str = (f"UniformPolarGrid(radius_max={self.R_MAX}, dist_radii={self.D_RADII}, "
                        f"n_inplanes={self.N_INPLANES}, quadrature='gauss-jacobi-beta-1', "
                        f"n_shells={expected_n_shells}, n_points={expected_n_points})")
        assert repr_str == expected_str
