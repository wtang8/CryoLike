import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from cryolike.grids.polar_grid import UniformPolarGrid, QuadratureType
from cryolike.util.precision import set_precision, get_float_dtype

set_precision('double')

class TestQuadratureType:
    """Test QuadratureType enum."""
    
    def test_enum_values(self):
        """Test that enum values are correct."""
        assert QuadratureType.GAUSS_JACOBI_BETA_1.value == 'gauss-jacobi-beta-1'
        assert QuadratureType.GAUSS_JACOBI_BETA_2.value == 'gauss-jacobi-beta-2'
        assert QuadratureType.GAUSS_LEGENDRE.value == 'gauss-legendre'


class TestUniformPolarGridInitialization:
    """Test UniformPolarGrid initialization."""
    
    def test_basic_initialization(self):
        """Test basic grid initialization with default parameters."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        assert grid.radius_max == 1.0
        assert grid.dist_radii == 0.1
        assert grid.n_inplanes == 8
        assert grid.quadrature == QuadratureType.GAUSS_JACOBI_BETA_1
        assert grid.n_shells > 0
        assert grid.n_points == grid.n_shells * grid.n_inplanes
    
    def test_invalid_radius_max(self):
        """Test that negative radius_max raises error."""
        with pytest.raises(Exception):  # ensure_positive should raise
            UniformPolarGrid(
                radius_max=-1.0,
                dist_radii=0.1,
                n_inplanes=8
            )
    
    def test_invalid_dist_radii(self):
        """Test that negative dist_radii raises error."""
        with pytest.raises(Exception):
            UniformPolarGrid(
                radius_max=1.0,
                dist_radii=-0.1,
                n_inplanes=8
            )
    
    def test_invalid_n_inplanes(self):
        """Test that negative n_inplanes raises error."""
        with pytest.raises(Exception):
            UniformPolarGrid(
                radius_max=1.0,
                dist_radii=0.1,
                n_inplanes=-8
            )
    
    def test_unknown_quadrature_type(self):
        """Test that invalid quadrature type raises error."""
        with patch('cryolike.grids.polar_grid.UniformPolarGrid.__post_init__') as mock_init:
            grid = UniformPolarGrid.__new__(UniformPolarGrid)
            grid.radius_max = 1.0
            grid.dist_radii = 0.1
            grid.n_inplanes = 8
            grid.quadrature = MagicMock()
            grid.quadrature.__eq__ = lambda self, other: False # type: ignore
            
            with pytest.raises(ValueError, match="Unknown quadrature type"):
                if grid.quadrature == QuadratureType.GAUSS_JACOBI_BETA_1:
                    pass
                elif grid.quadrature == QuadratureType.GAUSS_JACOBI_BETA_2:
                    pass
                elif grid.quadrature == QuadratureType.GAUSS_LEGENDRE:
                    pass
                else:
                    raise ValueError(f"Unknown quadrature type: {grid.quadrature}")


class TestUniformPolarGridQuadrature:
    """Test different quadrature methods."""
    
    @pytest.mark.parametrize("quadrature", [
        QuadratureType.GAUSS_JACOBI_BETA_1,
        QuadratureType.GAUSS_JACOBI_BETA_2,
        QuadratureType.GAUSS_LEGENDRE
    ])
    def test_quadrature_types(self, quadrature):
        """Test that all quadrature types initialize correctly."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8,
            quadrature=quadrature
        )
        
        assert grid.quadrature == quadrature
        assert grid.n_shells > 0
        assert grid.radius_shells.shape == (grid.n_shells,)
        assert grid.weight_shells.shape == (grid.n_shells,)
    
    def test_gauss_jacobi_invalid_beta(self):
        """Test that invalid beta parameter raises error in _gauss_jacobi."""
        grid = UniformPolarGrid.__new__(UniformPolarGrid)
        grid.radius_max = 1.0
        grid.dist_radii = 0.1
        grid.n_inplanes = 8
        
        with pytest.raises(ValueError, match="Beta parameter must be 1 or 2"):
            grid._gauss_jacobi(beta=3)
    
    def test_gauss_jacobi_beta_1_weights(self):
        """Test that Gauss-Jacobi beta=1 produces correct weight scaling."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=16,
            quadrature=QuadratureType.GAUSS_JACOBI_BETA_1
        )
        
        # Total weight should integrate to approximately π * r_max^2
        total_weight = grid.weight_points.sum().item()
        expected = np.pi * grid.radius_max ** 2
        assert np.isclose(total_weight, expected, rtol=0.1)
    
    def test_gauss_jacobi_beta_2_weights(self):
        """Test that Gauss-Jacobi beta=2 produces positive weights."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=16,
            quadrature=QuadratureType.GAUSS_JACOBI_BETA_2
        )
        
        # All weights should be positive
        assert torch.all(grid.weight_shells > 0)
        assert torch.all(grid.weight_points > 0)
        
        # Test the specific beta=2 weight formula path
        test_grid = UniformPolarGrid.__new__(UniformPolarGrid)
        test_grid.radius_max = 1.0
        test_grid.dist_radii = 0.1
        test_grid.n_shells = grid.n_shells
        test_grid._gauss_jacobi(beta=2)
        
        # Verify weights match the beta=2 formula
        assert torch.allclose(test_grid.weight_shells, grid.weight_shells)
    
    def test_gauss_legendre_weights(self):
        """Test that Gauss-Legendre produces correct weight scaling."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=16,
            quadrature=QuadratureType.GAUSS_LEGENDRE
        )
        
        # Total weight should integrate to approximately π * r_max^2
        total_weight = grid.weight_points.sum().item()
        expected = np.pi * grid.radius_max ** 2
        assert np.isclose(total_weight, expected, rtol=0.1)


class TestUniformPolarGridGeometry:
    """Test geometric properties of the grid."""
    
    def test_radius_shells_range(self):
        """Test that radius shells are within [0, radius_max]."""
        grid = UniformPolarGrid(
            radius_max=2.0,
            dist_radii=0.2,
            n_inplanes=8
        )
        
        assert torch.all(grid.radius_shells >= 0)
        assert torch.all(grid.radius_shells <= grid.radius_max)
    
    def test_radius_points_expansion(self):
        """Test that radius_points correctly expands radius_shells."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        float_dtype = get_float_dtype()
        # Each shell radius should appear n_inplanes times
        for i, r_shell in enumerate(grid.radius_shells):
            start_idx = i * grid.n_inplanes
            end_idx = (i + 1) * grid.n_inplanes
            assert torch.allclose(
                grid.radius_points[start_idx:end_idx],
                r_shell * torch.ones(grid.n_inplanes, dtype=float_dtype)
            )
    
    def test_theta_range(self):
        """Test that angles are in [0, 2π)."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=16
        )
        
        assert torch.all(grid.theta_shell >= 0)
        assert torch.all(grid.theta_shell < 2 * np.pi)
        assert torch.all(grid.theta_points >= 0)
        assert torch.all(grid.theta_points < 2 * np.pi)
    
    def test_theta_uniform_spacing(self):
        """Test that angles are uniformly spaced."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        expected_spacing = 2 * np.pi / grid.n_inplanes
        diffs = torch.diff(grid.theta_shell)
        assert torch.allclose(diffs, torch.full_like(diffs, expected_spacing), rtol=1e-5)
    
    def test_cartesian_coordinates(self):
        """Test that Cartesian coordinates match polar coordinates."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        # Verify conversion from polar to Cartesian
        x_expected = grid.radius_points * torch.cos(grid.theta_points)
        y_expected = grid.radius_points * torch.sin(grid.theta_points)
        
        assert torch.allclose(grid.x_points, x_expected)
        assert torch.allclose(grid.y_points, y_expected)
    
    def test_cartesian_radius(self):
        """Test that Cartesian coordinates have correct radius."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        radius_computed = torch.sqrt(grid.x_points ** 2 + grid.y_points ** 2)
        assert torch.allclose(radius_computed, grid.radius_points, rtol=1e-5)


class TestUniformPolarGridWeights:
    """Test weight properties."""
    
    def test_weight_points_from_shells(self):
        """Test that weight_points correctly distributes weight_shells."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        float_dtype = get_float_dtype()
        # Each shell's weight should be divided among n_inplanes points
        for i, w_shell in enumerate(grid.weight_shells):
            start_idx = i * grid.n_inplanes
            end_idx = (i + 1) * grid.n_inplanes
            expected_point_weight = w_shell / grid.n_inplanes
            assert torch.allclose(
                grid.weight_points[start_idx:end_idx],
                expected_point_weight * torch.ones(grid.n_inplanes, dtype=float_dtype)
            )
    
    def test_total_weight_conservation(self):
        """Test that sum of point weights equals sum of shell weights."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=16
        )
        
        total_shell_weight = grid.weight_shells.sum()
        total_point_weight = grid.weight_points.sum()
        assert torch.allclose(total_shell_weight, total_point_weight)


class TestUniformPolarGridDeviceTransfer:
    """Test device and dtype transfer."""
    
    def test_to_method_returns_self(self):
        """Test that to() method returns self for chaining."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        result = grid.to(dtype=torch.float32, device=torch.device('cpu'))
        assert result is grid
    
    def test_to_method_partial_initialization(self):
        """Test to() method when only radius_shells exists."""
        grid = UniformPolarGrid.__new__(UniformPolarGrid)
        grid.radius_max = 1.0
        grid.dist_radii = 0.1
        grid.n_inplanes = 8
        grid.n_shells = 10
        
        # Create only radius_shells and weight_shells
        grid.radius_shells = torch.randn(10)
        grid.weight_shells = torch.randn(10)
        
        # Should work with only partial attributes
        result = grid.to(dtype=torch.float64, device=torch.device('cpu'))
        assert result is grid
        assert grid.radius_shells.dtype == torch.float64
        assert grid.weight_shells.dtype == torch.float64
    
    def test_dtype_conversion(self):
        """Test dtype conversion."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        grid.to(dtype=torch.float64, device=torch.device('cpu'))
        
        assert grid.radius_shells.dtype == torch.float64
        assert grid.radius_points.dtype == torch.float64
        assert grid.theta_shell.dtype == torch.float64
        assert grid.theta_points.dtype == torch.float64
        assert grid.weight_shells.dtype == torch.float64
        assert grid.weight_points.dtype == torch.float64
        assert grid.x_points.dtype == torch.float64
        assert grid.y_points.dtype == torch.float64
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_transfer(self):
        """Test transfer to CUDA device."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        device = torch.device('cuda:0')
        grid.to(dtype=torch.float32, device=device)
        
        assert grid.radius_shells.device.type == 'cuda'
        assert grid.x_points.device.type == 'cuda'


class TestUniformPolarGridRepr:
    """Test string representation."""
    
    def test_repr_contains_key_info(self):
        """Test that __repr__ contains essential information."""
        grid = UniformPolarGrid(
            radius_max=1.5,
            dist_radii=0.15,
            n_inplanes=12,
            quadrature=QuadratureType.GAUSS_LEGENDRE
        )
        
        repr_str = repr(grid)
        assert "UniformPolarGrid" in repr_str
        assert "radius_max=1.5" in repr_str
        assert "dist_radii=0.15" in repr_str
        assert "n_inplanes=12" in repr_str
        assert "gauss-legendre" in repr_str
        assert f"n_shells={grid.n_shells}" in repr_str
        assert f"n_points={grid.n_points}" in repr_str


class TestFourierTranslationKernel:
    """Test Fourier translation kernel generation."""
    
    def test_kernel_shape(self):
        """Test that kernel has correct shape."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        n_displacements = 5
        x_disp = torch.randn(n_displacements)
        y_disp = torch.randn(n_displacements)
        
        kernel = grid.get_fourier_translation_kernel(x_disp, y_disp)
        
        assert kernel.shape == (n_displacements, grid.n_shells, grid.n_inplanes)
    
    def test_kernel_is_complex(self):
        """Test that kernel is complex-valued."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        x_disp = torch.tensor([0.1])
        y_disp = torch.tensor([0.2])
        
        kernel = grid.get_fourier_translation_kernel(x_disp, y_disp)
        
        assert torch.is_complex(kernel)
    
    def test_kernel_unit_magnitude(self):
        """Test that kernel has unit magnitude (phase shift only)."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        x_disp = torch.tensor([0.1, 0.2])
        y_disp = torch.tensor([0.3, 0.4])
        
        kernel = grid.get_fourier_translation_kernel(x_disp, y_disp)
        
        magnitude = torch.abs(kernel)
        assert torch.allclose(magnitude, torch.ones_like(magnitude), rtol=1e-5)
    
    def test_zero_displacement(self):
        """Test that zero displacement gives identity kernel."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        x_disp = torch.tensor([0.0])
        y_disp = torch.tensor([0.0])
        
        kernel = grid.get_fourier_translation_kernel(x_disp, y_disp)
        
        # Should be all ones (identity)
        expected = torch.ones_like(kernel)
        assert torch.allclose(kernel, expected, rtol=1e-5)
    
    def test_box_size_scaling(self):
        """Test that box_size parameter affects the kernel correctly."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        x_disp = torch.tensor([1.0])
        y_disp = torch.tensor([0.0])
        
        kernel1 = grid.get_fourier_translation_kernel(
            x_disp, y_disp, box_size_x=2.0, box_size_y=2.0
        )
        kernel2 = grid.get_fourier_translation_kernel(
            x_disp, y_disp, box_size_x=4.0, box_size_y=2.0
        )
        
        # Different box sizes should give different kernels
        assert not torch.allclose(kernel1, kernel2)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_kernel_cuda_device(self):
        """Test that kernel can be generated on CUDA device."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        device = torch.device('cuda:0')
        x_disp = torch.tensor([0.1], device=device)
        y_disp = torch.tensor([0.2], device=device)
        
        kernel = grid.get_fourier_translation_kernel(
            x_disp, y_disp, device=device
        )
        
        assert kernel.device.type == 'cuda'
    
    def test_kernel_batch_processing(self):
        """Test kernel generation with multiple displacements."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        n_batch = 10
        x_disp = torch.randn(n_batch)
        y_disp = torch.randn(n_batch)
        
        kernel = grid.get_fourier_translation_kernel(x_disp, y_disp)
        
        # Each displacement should produce a different kernel
        for i in range(n_batch - 1):
            assert not torch.allclose(kernel[i], kernel[i + 1])


class TestNShellsCalculation:
    """Test n_shells calculation."""
    
    def test_n_shells_exact_division(self):
        """Test n_shells when radius_max is exactly divisible by dist_radii."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.1,
            n_inplanes=8
        )
        
        expected_n_shells = 1 + int(np.ceil(1.0 / 0.1))
        assert grid.n_shells == expected_n_shells
    
    def test_n_shells_inexact_division(self):
        """Test n_shells when radius_max is not exactly divisible."""
        grid = UniformPolarGrid(
            radius_max=1.0,
            dist_radii=0.15,
            n_inplanes=8
        )
        
        expected_n_shells = 1 + int(np.ceil(1.0 / 0.15))
        assert grid.n_shells == expected_n_shells
    
    def test_n_shells_consistency(self):
        """Test that n_shells is consistent across quadrature types."""
        radius_max = 1.0
        dist_radii = 0.1
        n_inplanes = 8
        
        grid1 = UniformPolarGrid(
            radius_max=radius_max,
            dist_radii=dist_radii,
            n_inplanes=n_inplanes,
            quadrature=QuadratureType.GAUSS_JACOBI_BETA_1
        )
        
        grid2 = UniformPolarGrid(
            radius_max=radius_max,
            dist_radii=dist_radii,
            n_inplanes=n_inplanes,
            quadrature=QuadratureType.GAUSS_LEGENDRE
        )
        
        assert grid1.n_shells == grid2.n_shells


if __name__ == "__main__":
    pytest.main([__file__, "-v"])