import unittest
import torch
import numpy as np

from cryolike.pose import Displacements2D
from cryolike.grid import UniformPolarGrid, SquaredCartesianGrid2D
from cryolike.stack import PhysicalImages, FourierImages

class TestDisplacements2D(unittest.TestCase):

    def setUp(self):
        self.x_displacements_angstrom = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        self.y_displacements_angstrom = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        self.box_size_angstrom = 100.0
        self.displacements = Displacements2D(
            x_displacements_angstrom=self.x_displacements_angstrom,
            y_displacements_angstrom=self.y_displacements_angstrom,
            box_size_angstrom=self.box_size_angstrom
        )
        self.polar_grid = UniformPolarGrid(
            radius_max=10.0,
            dist_radii=1.0,
            n_inplanes=8
        )

    def test_initialization(self):
        """Test successful initialization and correct attribute values."""
        self.assertTrue(torch.equal(self.displacements.x_displacements_angstrom, self.x_displacements_angstrom))
        self.assertTrue(torch.equal(self.displacements.y_displacements_angstrom, self.y_displacements_angstrom))
        self.assertEqual(self.displacements.box_size_angstrom, self.box_size_angstrom)

    def test_initialization_failures(self):
        """Test initialization with invalid arguments."""
        # Wrong dimension for x_displacements
        with self.assertRaisesRegex(AssertionError, "x_displacements must be a 1D array"):
            Displacements2D(torch.randn(2, 2), self.y_displacements_angstrom, self.box_size_angstrom)

        # Wrong dimension for y_displacements
        with self.assertRaisesRegex(AssertionError, "y_displacements must be a 1D array"):
            Displacements2D(self.x_displacements_angstrom, torch.randn(2, 2), self.box_size_angstrom)

        # Shape mismatch
        with self.assertRaisesRegex(AssertionError, "x_displacements and y_displacements must have the same shape"):
            Displacements2D(torch.randn(3), torch.randn(4), self.box_size_angstrom)

        # Dtype mismatch
        with self.assertRaisesRegex(AssertionError, "x_displacements and y_displacements must have the same dtype"):
            Displacements2D(torch.randn(3, dtype=torch.float32), torch.randn(3, dtype=torch.float64), self.box_size_angstrom)

    def test_xy_displacements_angstrom(self):
        """Test the xy_displacements_angstrom property."""
        expected_xy_displacements = torch.stack((self.x_displacements_angstrom, self.y_displacements_angstrom), dim=-1)
        self.assertTrue(torch.equal(self.displacements.xy_displacements_angstrom, expected_xy_displacements))

    def test_n_displacements(self):
        """Test the n_displacements property."""
        self.assertEqual(self.displacements.n_displacements, self.x_displacements_angstrom.shape[0])

    def test_to_method(self):
        """Test the 'to' method for changing dtype and device."""
        # Test dtype conversion
        disp_float32 = self.displacements.to(dtype=torch.float32, device='cpu')
        self.assertEqual(disp_float32.x_displacements_angstrom.dtype, torch.float32)
        self.assertEqual(disp_float32.y_displacements_angstrom.dtype, torch.float32)

        # Test device conversion if CUDA is available
        if torch.cuda.is_available():
            disp_cuda = self.displacements.to(dtype=torch.float64, device='cuda')
            self.assertTrue(disp_cuda.x_displacements_angstrom.is_cuda)
            self.assertTrue(disp_cuda.y_displacements_angstrom.is_cuda)

    def test_kernel_calculation(self):
        """Test the kernel calculation."""
        kernel = self.displacements.kernel(self.polar_grid)

        # Check shape
        expected_shape = (
            self.displacements.x_displacements_angstrom.shape[0],
            self.polar_grid.n_shells,
            self.polar_grid.n_inplanes
        )
        self.assertEqual(kernel.shape, expected_shape)

        # Check values
        x_pts = self.polar_grid.x_points[None, :, :]
        y_pts = self.polar_grid.y_points[None, :, :]
        x_disp = self.displacements.x_displacements_angstrom[:, None, None]
        y_disp = self.displacements.y_displacements_angstrom[:, None, None]

        expected_kernel = torch.exp(
            - (2.0 * np.pi / self.box_size_angstrom * 2.0) * 1j * (
                x_disp * x_pts + y_disp * y_pts
            )
        )

        self.assertTrue(torch.allclose(kernel, expected_kernel))

    def test_kernel_with_single_displacement(self):
        """Test kernel calculation with a single displacement."""
        disp = Displacements2D(
            x_displacements_angstrom=torch.tensor([1.0]),
            y_displacements_angstrom=torch.tensor([2.0]),
            box_size_angstrom=self.box_size_angstrom
        )
        kernel = disp.kernel(self.polar_grid)
        expected_shape = (1, self.polar_grid.n_shells, self.polar_grid.n_inplanes)
        self.assertEqual(kernel.shape, expected_shape)

    def test_kernel_with_fourier_transform(self):
        """Test kernel calculation with a Fourier transform."""
        sigma = 2.0
        n_pixels = 128
        phys_grid = SquaredCartesianGrid2D(n_pixels, self.box_size_angstrom)
        images_phys = torch.exp(- (phys_grid.x_axis[:,None] ** 2 + phys_grid.y_axis[None,:] ** 2) / (2 * sigma ** 2)).unsqueeze(0)
        physical_images = PhysicalImages(phys_grid, images_phys)
        fourier_images = physical_images.transform_to_fourier(self.polar_grid)
        disp = Displacements2D(
            x_displacements_angstrom=self.x_displacements_angstrom,
            y_displacements_angstrom=self.y_displacements_angstrom,
            box_size_angstrom=self.box_size_angstrom
        )
        kernel = disp.kernel(self.polar_grid)
        fourier_images.images_fourier  = fourier_images.images_fourier * kernel
        images_phys_true = torch.exp(- (
            (phys_grid.x_axis[None,:] - self.x_displacements_angstrom[:,None])[:,:,None] ** 2 +
            (phys_grid.y_axis[None,:] - self.y_displacements_angstrom[:,None])[:,None,:] ** 2
        ) / (2 * sigma ** 2))
        physical_images_true = PhysicalImages(phys_grid, images_phys_true)
        fourier_images_true = physical_images_true.transform_to_fourier(self.polar_grid)
        
        cross_correlation = (self.polar_grid.integrate(
            fourier_images.images_fourier.conj() * fourier_images_true.images_fourier
        ) / torch.sqrt(
            self.polar_grid.integrate(
                fourier_images.images_fourier.conj() * fourier_images.images_fourier
            ) * self.polar_grid.integrate(
                fourier_images_true.images_fourier.conj() * fourier_images_true.images_fourier
            )
        )).real

        assert torch.allclose(cross_correlation, torch.ones_like(cross_correlation), atol=1e-3)

    def test_sample_grid(self):
        """Test the sample_grid method."""
        max_displacements_angstrom = 10.0
        n_samples_per_axis = 5

        displacements = Displacements2D.sample_grid(
            max_displacements_angstrom=max_displacements_angstrom,
            n_samples_per_axis=n_samples_per_axis,
            box_size_angstrom=self.box_size_angstrom
        )

        self.assertEqual(displacements.box_size_angstrom, self.box_size_angstrom)
        self.assertEqual(displacements.x_displacements_angstrom.shape[0], n_samples_per_axis ** 2)
        self.assertEqual(displacements.y_displacements_angstrom.shape[0], n_samples_per_axis ** 2)

        _x_axis = np.linspace(
            - max_displacements_angstrom, 
              max_displacements_angstrom, 
            n_samples_per_axis, endpoint=True)
        _y_axis = _x_axis
        x_disp_expected, y_disp_expected = np.meshgrid(_x_axis, _y_axis, indexing='ij')
        x_disp_expected = torch.from_numpy(x_disp_expected.flatten())
        y_disp_expected = torch.from_numpy(y_disp_expected.flatten())

        self.assertTrue(torch.allclose(displacements.x_displacements_angstrom, x_disp_expected))
        self.assertTrue(torch.allclose(displacements.y_displacements_angstrom, y_disp_expected))

    def test_sample_grid_n_sample_one(self):
        """Test sample grid with n_samples_per_axis = 1."""
        displacements = Displacements2D.sample_grid(
            max_displacements_angstrom=10.0,
            n_samples_per_axis=1,
            box_size_angstrom=self.box_size_angstrom
        )
        self.assertEqual(displacements.box_size_angstrom, self.box_size_angstrom)
        self.assertTrue(torch.allclose(displacements.x_displacements_angstrom, torch.zeros(1)))
        self.assertTrue(torch.allclose(displacements.y_displacements_angstrom, torch.zeros(1)))


if __name__ == '__main__':
    unittest.main()