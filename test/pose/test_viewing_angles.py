import unittest
import torch
from unittest.mock import patch, MagicMock
from cryolike.util import PrecisionLevel
from cryolike.pose.viewing_angles import ViewingAngles

DEVICES = [torch.device('cpu'), torch.device('cuda')] if torch.cuda.is_available() else [torch.device('cpu')]

class TestViewingAngles(unittest.TestCase):

    def test_init_full_params(self):
        """Tests initialization with all parameters provided."""
        n_angles = 10
        azimus = torch.rand(n_angles, dtype=torch.float32)
        polars = torch.rand(n_angles, dtype=torch.float32)
        gammas = torch.rand(n_angles, dtype=torch.float32)
        weights = torch.rand(n_angles, dtype=torch.float32)

        va = ViewingAngles(
            azimus=azimus,
            polars=polars,
            gammas=gammas, 
            weights_viewing=weights
        )

        self.assertTrue(torch.equal(va.azimus, azimus))
        self.assertTrue(torch.equal(va.polars, polars))
        self.assertTrue(torch.equal(va.gammas, gammas))
        self.assertTrue(torch.equal(va.weights_viewing, weights))
        self.assertEqual(va.n_angles, n_angles)

    def test_init_defaults(self):
        """Tests default initialization for gammas and weights_viewing."""
        n_angles = 5
        azimus = torch.rand(n_angles, dtype=torch.float32)
        polars = torch.rand(n_angles, dtype=torch.float32)

        va = ViewingAngles(
            azimus=azimus,
            polars=polars
        )

        # Check default gammas
        expected_gammas = torch.zeros_like(azimus)
        self.assertTrue(torch.equal(va.gammas, expected_gammas))

        # Check default weights
        expected_weights = torch.ones_like(azimus) / n_angles
        self.assertTrue(torch.allclose(va.weights_viewing, expected_weights))

        self.assertEqual(va.n_angles, n_angles)

    def test_precision_property(self):
        """Tests the precision property for both single and double dtypes."""
        # Test SINGLE precision
        azimus_32 = torch.rand(5, dtype=torch.float32)
        polars_32 = torch.rand(5, dtype=torch.float32)
        va_single = ViewingAngles(azimus=azimus_32, polars=polars_32)
        self.assertEqual(va_single.precision, PrecisionLevel.SINGLE)

        # Test DOUBLE precision
        azimus_64 = torch.rand(5, dtype=torch.float64)
        polars_64 = torch.rand(5, dtype=torch.float64)
        va_double = ViewingAngles(azimus=azimus_64, polars=polars_64)
        self.assertEqual(va_double.precision, PrecisionLevel.DOUBLE)

    def test_init_dtype_mismatch(self):
        n_angles = 4
        azimus = torch.rand(n_angles, dtype=torch.float32)
        polars_wrong_dtype = torch.rand(n_angles, dtype=torch.float64)

        with self.assertRaises(AssertionError):
            va = ViewingAngles(
                azimus=azimus,
                polars=polars_wrong_dtype,
            )
        

    def test_post_init_shape_assertion(self):
        """Tests that __post_init__ raises AssertionError for shape mismatch."""
        n_angles = 4
        azimus = torch.rand(n_angles, dtype=torch.float32)
        # polars has a different shape
        polars_wrong_shape = torch.rand(n_angles + 1, dtype=torch.float32)

        with self.assertRaises(AssertionError):
            ViewingAngles(
                azimus=azimus,
                polars=polars_wrong_shape,
            )

    @patch('cryolike.pose.viewing_angles.SphereShell')
    def test_from_viewing_distance(self, mock_sphere_shell):
        """Tests the from_viewing_distance class method."""
        # --- Setup Mock Data ---
        n_points = 20
        mock_azimu_points = torch.rand(n_points, dtype=torch.float64)
        mock_polar_points = torch.rand(n_points, dtype=torch.float64)
        mock_weight_points = torch.rand(n_points, dtype=torch.float64)

        # --- Configure Mock ---
        mock_shell_instance = MagicMock()
        mock_shell_instance.azimu_points = mock_azimu_points
        mock_shell_instance.polar_points = mock_polar_points
        mock_shell_instance.weight_points = mock_weight_points
        mock_sphere_shell.return_value = mock_shell_instance

        # --- Call Function ---
        viewing_distance = 0.1
        va = ViewingAngles.from_viewing_distance(viewing_distance, precision=PrecisionLevel.SINGLE)

        # --- Assertions ---
        # 1. SphereShell was called correctly
        mock_sphere_shell.assert_called_once_with(
            radius=1.0, dist_eq=viewing_distance, uniform_azimuthal_sampling=False, store_cartesian_points=False
        )

        # 2. ViewingAngles was initialized with the correct data (and correct dtype)
        self.assertEqual(va.azimus.dtype, torch.float32)
        self.assertEqual(va.polars.dtype, torch.float32)
        self.assertEqual(va.weights_viewing.dtype, torch.float32)
        self.assertTrue(torch.equal(va.gammas, torch.zeros_like(va.azimus)))
        self.assertEqual(va.n_angles, n_points)

    @patch('cryolike.pose.viewing_angles.SphereShell')
    def test_from_viewing_distance_no_precision(self, mock_sphere_shell):
        """Tests from_viewing_distance when precision is not specified."""
        # --- Setup Mock Data ---
        n_points = 15
        mock_azimu_points = torch.rand(n_points, dtype=torch.float64)
        mock_polar_points = torch.rand(n_points, dtype=torch.float64)
        mock_weight_points = torch.rand(n_points, dtype=torch.float64)

        # --- Configure Mock ---
        mock_shell_instance = MagicMock()
        mock_shell_instance.azimu_points = mock_azimu_points
        mock_shell_instance.polar_points = mock_polar_points
        mock_shell_instance.weight_points = mock_weight_points
        mock_sphere_shell.return_value = mock_shell_instance

        # --- Call Function ---
        viewing_distance = 0.2
        # Call without the precision argument
        va = ViewingAngles.from_viewing_distance(viewing_distance)

        # --- Assertions ---
        # SphereShell provides float64 tensors, so the resulting object should have float64 tensors
        self.assertEqual(va.azimus.dtype, torch.float64)

    def test_to_method_dtype_conversion(self):
        """Tests the to() method for changing dtype and device."""
        n_angles = 10
        azimus = torch.rand(n_angles, dtype=torch.float32)
        polars = torch.rand(n_angles, dtype=torch.float32)

        va = ViewingAngles(
            azimus=azimus,
            polars=polars,
        )

        # Convert to double precision
        va.to(dtype=torch.float64)

        self.assertEqual(va.azimus.dtype, torch.float64)
        self.assertEqual(va.polars.dtype, torch.float64)
        self.assertEqual(va.gammas.dtype, torch.float64)
        self.assertEqual(va.weights_viewing.dtype, torch.float64)

    def test_to_method_no_args(self):
        """Tests that calling .to() with no arguments does not change the object."""
        n_angles = 5
        azimus = torch.rand(n_angles, dtype=torch.float32)
        polars = torch.rand(n_angles, dtype=torch.float32)
        va = ViewingAngles(azimus=azimus, polars=polars)

        # Keep a copy of the original tensors
        original_azimus = va.azimus.clone()

        # Call .to() with no arguments
        va.to()

        # Check that the dtype and data have not changed
        self.assertEqual(va.azimus.dtype, torch.float32)
        self.assertTrue(torch.equal(va.azimus, original_azimus))


# Dynamically generate tests for each device to ensure compatibility with unittest.TestCase
for _idx, _device in enumerate(DEVICES):
    def make_test_func(device_val):
        def test_method(self):
            """Dynamically generated test for device conversion."""
            n_angles = 10
            azimus = torch.rand(n_angles, dtype=torch.float32)
            polars = torch.rand(n_angles, dtype=torch.float32)

            va = ViewingAngles(
                azimus=azimus,
                polars=polars,
            )

            va.to(device=device_val)

            self.assertEqual(va.azimus.device, device_val)
            self.assertEqual(va.polars.device, device_val)
            self.assertEqual(va.gammas.device, device_val)
            self.assertEqual(va.weights_viewing.device, device_val)
        return test_method

    setattr(TestViewingAngles, f"test_to_method_device_conversion_{_device.type}", make_test_func(_device))

if __name__ == '__main__':
    unittest.main()

            