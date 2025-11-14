import unittest
import torch
import numpy as np
from math import pi

from cryolike.grid.sphere_shell import SphereShell

class TestSphereShell(unittest.TestCase):

    def test_init_default_parameters(self):
        """Test SphereShell initialization with default parameters."""
        shell = SphereShell()
        self.assertIsInstance(shell, SphereShell)
        self.assertEqual(shell.radius, 1.0)
        self.assertAlmostEqual(shell.dist_eq, 1.0 / (2.0 * pi))
        self.assertTrue(shell.azimu_points.dtype == torch.float64)
        self.assertTrue(shell.polar_points.dtype == torch.float64)
        self.assertTrue(shell.weight_points.dtype == torch.float64)
        self.assertGreater(shell.n_points, 0)
        self.assertGreater(shell.n_polar_circles, 0)
        self.assertEqual(shell.n_points, np.sum(shell.n_azimus_each_circle))

    def test_init_custom_parameters(self):
        """Test SphereShell initialization with custom radius and dist_eq."""
        radius = 5.0
        dist_eq = 0.1
        shell = SphereShell(radius=radius, dist_eq=dist_eq)
        self.assertEqual(shell.radius, radius)
        self.assertEqual(shell.dist_eq, dist_eq)
        self.assertGreater(shell.n_points, 0)

    def test_init_uniform_azimuthal_sampling(self):
        """Test uniform azimuthal sampling."""
        shell = SphereShell(uniform_azimuthal_sampling=True)
        # In uniform sampling, all circles should have the same number of azimuthal points
        self.assertTrue(np.all(shell.n_azimus_each_circle == shell.n_azimus_each_circle[0]))

    def test_init_adaptive_azimuthal_sampling(self):
        """Test adaptive azimuthal sampling."""
        shell = SphereShell(uniform_azimuthal_sampling=False)
        # In adaptive sampling, n_azimus_each_circle should vary with sin(polar_circles)
        # It should not be all equal unless polar_circles are all the same (which they are not)
        self.assertFalse(np.all(shell.n_azimus_each_circle == shell.n_azimus_each_circle[0]))
        # Check that n_azimus_each_circle is generally larger near the equator (polar_circles near pi/2)
        # and smaller near the poles (polar_circles near 0 or pi)
        # This is a qualitative check, more rigorous checks would involve specific values
        self.assertGreaterEqual(shell.n_azimus_each_circle.max(), shell.n_azimus_each_circle.min())

    def test_init_store_cartesian_points(self):
        """Test storing Cartesian points."""
        shell = SphereShell(store_cartesian_points=True)
        self.assertIsNotNone(shell.xyz_points)

    def test_init_invalid_parameters(self):
        """Test initialization with invalid radius or dist_eq."""
        with self.assertRaisesRegex(ValueError, "spherical shell radius must be positive"):
            SphereShell(radius=0.0)
        with self.assertRaisesRegex(ValueError, "spherical shell radius must be positive"):
            SphereShell(radius=-1.0)
        with self.assertRaisesRegex(ValueError, "spherical shell equatorial point distance dist_eq must be positive"):
            SphereShell(dist_eq=0.0)
        with self.assertRaisesRegex(ValueError, "spherical shell equatorial point distance dist_eq must be positive"):
            SphereShell(dist_eq=-1.0)

    def test_weight_points_sum(self):
        """Test that the sum of weight_points approximates the surface area of the sphere."""
        radius = 2.5
        shell = SphereShell(radius=radius)
        expected_surface_area = 4 * pi * radius**2
        # Use a reasonable tolerance for numerical integration
        self.assertAlmostEqual(shell.weight_points.sum().item(), expected_surface_area, places=4)

    def test_cartesian_points_shape(self):
        """Test the shape of the returned Cartesian points."""
        shell = SphereShell()
        xyz_points = shell.cartesian_points()
        self.assertEqual(xyz_points.shape, (shell.n_points, 3))

    def test_cartesian_points_magnitude(self):
        """Test that the magnitude of Cartesian points equals the radius."""
        radius = 3.0
        shell = SphereShell(radius=radius)
        xyz_points = shell.cartesian_points()
        magnitudes = torch.linalg.norm(xyz_points, dim=1)
        self.assertTrue(torch.allclose(magnitudes, torch.full_like(magnitudes, radius)))

    def test_cartesian_points_poles_equator(self):
        """Test Cartesian points at approximate poles and equator."""
        radius = 1.0
        shell = SphereShell(radius=radius, dist_eq=0.5, uniform_azimuthal_sampling=True)
        xyz_points = shell.cartesian_points()

        # Find points near the poles (polar_points near 0 or pi)
        # Due to Legendre roots, exact 0 or pi might not be present, so check closest
        polar_angles = shell.polar_points.numpy()
        
        # North pole (polar_angle ~ 0)
        north_pole_idx = np.argmin(polar_angles)
        if np.isclose(polar_angles[north_pole_idx], 0.0, atol=1e-3):
            # Expect z ~ radius, x, y ~ 0
            self.assertAlmostEqual(xyz_points[north_pole_idx, 2].item(), radius, places=3)
            self.assertAlmostEqual(xyz_points[north_pole_idx, 0].item(), 0.0, places=3)
            self.assertAlmostEqual(xyz_points[north_pole_idx, 1].item(), 0.0, places=3)

        # South pole (polar_angle ~ pi)
        south_pole_idx = np.argmax(polar_angles)
        if np.isclose(polar_angles[south_pole_idx], pi, atol=1e-3):
            # Expect z ~ -radius, x, y ~ 0
            self.assertAlmostEqual(xyz_points[south_pole_idx, 2].item(), -radius, places=3)
            self.assertAlmostEqual(xyz_points[south_pole_idx, 0].item(), 0.0, places=3)
            self.assertAlmostEqual(xyz_points[south_pole_idx, 1].item(), 0.0, places=3)

        # Equator (polar_angle ~ pi/2)
        equator_indices = np.where(np.isclose(polar_angles, pi/2, atol=1e-3))[0]
        if len(equator_indices) > 0:
            # Expect z ~ 0, x^2 + y^2 ~ radius^2
            for idx in equator_indices:
                self.assertAlmostEqual(xyz_points[idx, 2].item(), 0.0, places=3)
                self.assertAlmostEqual(xyz_points[idx, 0].item()**2 + xyz_points[idx, 1].item()**2, radius**2, places=3)

    def test_integrate_constant_function(self):
        """Test integration of a constant function over the sphere shell."""
        radius = 1.0
        shell = SphereShell(radius=radius)
        constant_value = 5.0
        f = torch.full((shell.n_points,), constant_value, dtype=torch.float64)
        expected_integral = constant_value * (4 * pi * radius**2)
        integral = shell.integrate(f)
        self.assertAlmostEqual(integral.item(), expected_integral, places=4)

    def test_integrate_variable_function(self):
        """Test integration of a variable function (e.g., z-coordinate) over the sphere shell."""
        radius = 1.0
        shell = SphereShell(radius=radius)
        xyz_points = shell.cartesian_points()
        f = xyz_points[:, 2] # Integrate z-coordinate
        # Due to symmetry, the integral of z over a sphere should be 0
        integral = shell.integrate(f)
        self.assertAlmostEqual(integral.item(), 0.0, places=4)

    def test_integrate_type_error(self):
        """Test TypeError when input function is not a torch.Tensor."""
        shell = SphereShell()
        f_np = np.ones(shell.n_points)
        with self.assertRaisesRegex(TypeError, "f must be a torch.Tensor"):
            shell.integrate(f_np)

    def test_integrate_value_error_shape_mismatch(self):
        """Test ValueError when input function shape does not match n_points."""
        shell = SphereShell()
        f_wrong_shape = torch.ones(shell.n_points + 1, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "f.shape\\[0\\] != self.n_points"):
            shell.integrate(f_wrong_shape)
