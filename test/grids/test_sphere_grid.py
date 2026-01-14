import unittest
import torch
import numpy as np
from math import pi
import pytest

from cryolike.grid import SphereGrid

class TestSphereGrid(unittest.TestCase):

    def test_init_defaults(self):
        """Test SphereGrid initialization with default parameters."""
        grid = SphereGrid()
        self.assertIsInstance(grid, SphereGrid)
        self.assertEqual(grid.radius_max, 1.0)
        self.assertAlmostEqual(grid.dist_eq, 1.0 / (2.0 * pi))
        self.assertTrue(grid.uniform_azimuthal_sampling)
        self.assertTrue(grid.equal_shell)
        self.assertTrue(grid.uniform_dist)

        self.assertGreater(grid.n_shells, 0)
        self.assertGreater(grid.n_points, 0)
        self.assertEqual(len(grid.shells), grid.n_shells)
        self.assertEqual(grid.radius_points.shape, (grid.n_points,))
        self.assertEqual(grid.polar_points.shape, (grid.n_points,))
        self.assertEqual(grid.azimu_points.shape, (grid.n_points,))
        self.assertEqual(grid.weight_points.shape, (grid.n_points,))

    def test_init_custom_parameters(self):
        """Test SphereGrid initialization with custom parameters."""
        grid = SphereGrid(radius_max=5.0, dist_eq=0.2, uniform_azimuthal_sampling=False)
        self.assertEqual(grid.radius_max, 5.0)
        self.assertEqual(grid.dist_eq, 0.2)
        self.assertFalse(grid.uniform_azimuthal_sampling)

    def test_init_invalid_parameters(self):
        """Test initialization with invalid (non-positive) parameters."""
        with self.assertRaisesRegex(ValueError, "maximum spherical radius must be positive"):
            SphereGrid(radius_max=0)
        with self.assertRaisesRegex(ValueError, "maximum spherical radius must be positive"):
            SphereGrid(radius_max=-10.0)
        with self.assertRaisesRegex(ValueError, "minimum equatorial distance must be positive"):
            SphereGrid(dist_eq=0)
        with self.assertRaisesRegex(ValueError, "minimum equatorial distance must be positive"):
            SphereGrid(dist_eq=-0.5)

    def test_equal_shell_true(self):
        """Test the 'equal_shell=True' configuration."""
        grid = SphereGrid(equal_shell=True)
        self.assertTrue(grid.equal_shell)
        
        # All shells should have the same number of points
        n_points_per_shell = grid.shells[0].n_points
        for shell in grid.shells:
            self.assertEqual(shell.n_points, n_points_per_shell)
        
        self.assertEqual(grid.n_points, grid.n_shells * n_points_per_shell)
        
        # Check that angular points repeat
        self.assertTrue(torch.allclose(
            grid.polar_points[:n_points_per_shell],
            grid.polar_points[n_points_per_shell : 2 * n_points_per_shell]
        ))
        self.assertTrue(torch.allclose(
            grid.azimu_points[:n_points_per_shell],
            grid.azimu_points[n_points_per_shell : 2 * n_points_per_shell]
        ))

    def test_equal_shell_false_adaptive_dist(self):
        """Test the 'equal_shell=False, uniform_dist=False' configuration."""
        grid = SphereGrid(equal_shell=False, uniform_dist=False, radius_max=2.0, dist_eq=0.5)
        self.assertFalse(grid.equal_shell)

        # distance between points per shell should vary
        for i in range(1, len(grid.shells)):
            self.assertGreater(grid.shells[i].dist_eq, grid.shells[i-1].dist_eq)

        # Total points should be the sum of points in each shell
        self.assertEqual(grid.n_points, sum([shell.n_points for shell in grid.shells]))

    def test_equal_shell_false_uniform_dist(self):
        """Test the 'equal_shell=False, uniform_dist=True' configuration."""
        grid = SphereGrid(equal_shell=False, uniform_dist=True, radius_max=2.0, dist_eq=0.5)
        self.assertFalse(grid.equal_shell)

        # distance between points per shell should be the same
        for i in range(1, len(grid.shells)):
            self.assertEqual(grid.shells[i].dist_eq, grid.shells[i-1].dist_eq)

        # Total points should be the sum of points in each shell
        self.assertEqual(grid.n_points, sum([shell.n_points for shell in grid.shells]))

    def test_point_properties(self):
        """Test properties of the grid points."""
        radius_max = 2.0
        grid = SphereGrid(radius_max=radius_max)

        # Radii should be within the max radius
        self.assertTrue(torch.all(grid.radius_points <= radius_max))
        self.assertTrue(torch.all(grid.radius_points >= 0))

        # Angles should be in the correct range
        self.assertTrue(torch.all(grid.polar_points >= 0) and torch.all(grid.polar_points <= pi))
        self.assertTrue(torch.all(grid.azimu_points >= 0) and torch.all(grid.azimu_points < 2 * pi))

    def test_integrate_constant_function_equal_shells(self):
        """Test integration of a constant function, which should yield constant * volume."""
        radius_max = 1.5
        grid = SphereGrid(equal_shell=True, radius_max=radius_max, dist_eq=0.2)
        
        constant_value = 5.0
        f = torch.full((grid.n_points,), constant_value, dtype=torch.float64)
        
        integral = grid.integrate(f)
        expected_volume = (4.0 / 3.0) * pi * (radius_max ** 3)
        expected_integral = constant_value * expected_volume
        
        # Numerical integration has tolerance
        self.assertAlmostEqual(integral.item(), expected_integral, places=3)

    def test_integrate_variable_function_equal_shells(self):
        """Test integration of f(x,y,z) = z. Due to symmetry, the result should be 0."""
        grid = SphereGrid(equal_shell=True, radius_max=2.0, dist_eq=0.3)
        
        # The SphereGrid doesn't automatically compute cartesian points, so we do it here
        z_points = grid.radius_points * torch.cos(grid.polar_points)
        
        integral = grid.integrate(z_points)
        self.assertAlmostEqual(integral.item(), 0.0, places=4)

    @pytest.mark.xfail
    def test_integrate_riesz_equal_shells(self):
        """Test Riesz integration of a constant function."""
        radius_max = 1.5
        grid = SphereGrid(equal_shell=True, radius_max=radius_max, dist_eq=0.05)
        
        constant_value = 5.0
        f = torch.full((grid.n_points,), constant_value, dtype=torch.float64)
        
        # Integral of c/r * r^2 dr d(omega) = c * integral of r dr d(omega)
        # = c * 2 * pi * r^2 | from 0 to R = c * 2 * pi * R^2
        integral = grid.integrate(f, use_riesz_integration=True)
        expected_integral = constant_value * 2 * pi * radius_max**2
        
        self.assertAlmostEqual(integral.item(), expected_integral, places=3)

    ## TODO: test integration on adaptive shells
    ## ...

    def test_integrate_errors(self):
        """Test that the integrate method raises appropriate errors."""
        grid = SphereGrid()
        
        # Test shape mismatch error
        f_wrong_shape = torch.ones(grid.n_points + 1)
        with self.assertRaisesRegex(ValueError, "f.shape\\[-1\\] != self.n_points"):
            grid.integrate(f_wrong_shape)
            
        # Test type error
        f_wrong_type = np.ones(grid.n_points)
        with self.assertRaises(AssertionError):
            grid.integrate(f_wrong_type) # type: ignore


if __name__ == '__main__':
    unittest.main()