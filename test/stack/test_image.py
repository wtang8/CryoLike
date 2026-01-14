import numpy as np
import torch
import pytest
from pytest import mark, raises
from unittest.mock import patch, MagicMock

from cryolike.grid import UniformPolarGrid, SquaredCartesianGrid2D, QuadratureType
from cryolike.stack.image import PhysicalImages, FourierImages
from cryolike.util import PrecisionLevel, get_complex_dtype

DEVICES = [torch.device("cpu"), torch.device("cuda")] if torch.cuda.is_available() else [torch.device("cpu")]
@pytest.fixture
def phys_grid_128():
    return SquaredCartesianGrid2D(n_pixels=128, box_size=128.0)

@pytest.fixture
def polar_grid_128():
    return UniformPolarGrid(radius_max=32, dist_radii=1.0, n_inplanes=128)

@pytest.fixture
def sample_phys_images(phys_grid_128):
    images = torch.randn(10, 128, 128)
    return PhysicalImages(phys_grid=phys_grid_128, images_phys=images)

@pytest.fixture
def sample_fourier_images(polar_grid_128):
    images = torch.randn(10, polar_grid_128.n_shells, polar_grid_128.n_inplanes, dtype=torch.complex64)
    return FourierImages(polar_grid=polar_grid_128, box_size=128.0, images_fourier=images)


class TestPhysicalImages:
    def test_post_init_raises_error_on_shape_mismatch(self, phys_grid_128):
        with raises(ValueError, match="Dimension mismatch"):
            PhysicalImages(phys_grid=phys_grid_128, images_phys=torch.randn(10, 64, 64))

    @patch('cryolike.stack.image.mrcfile.open')
    def test_from_mrc(self, mock_mrc_open):
        mock_mrc = MagicMock()
        mock_mrc.data = np.random.rand(5, 128, 128).astype(np.float32)
        mock_mrc_open.return_value.__enter__.return_value = mock_mrc

        images = PhysicalImages.from_mrc("test.mrc", pixel_size=1.0)
        assert images.n_images == 5
        assert images.n_pixels == 128
        assert images.box_size == 128.0
        assert torch.equal(images.images_phys, torch.from_numpy(mock_mrc.data))

    @patch('cryolike.stack.image.mrcfile.open')
    def test_from_mrc_2d_data(self, mock_mrc_open):
        mock_mrc = MagicMock()
        mock_mrc.data = np.random.rand(128, 128).astype(np.float32)
        mock_mrc_open.return_value.__enter__.return_value = mock_mrc

        images = PhysicalImages.from_mrc("test.mrcs", pixel_size=1.0)
        assert images.n_images == 1
        assert images.images_phys.shape == (1, 128, 128)

    def test_from_mrc_raises_error_on_invalid_extension(self):
        with raises(ValueError, match="Invalid file format"):
            PhysicalImages.from_mrc("test.txt", pixel_size=1.0)

    @patch('cryolike.stack.image.mrcfile.new')
    def test_save_to_mrc(self, mock_mrc_new, sample_phys_images):
        mock_mrc_file = MagicMock()
        mock_mrc_new.return_value.__enter__.return_value = mock_mrc_file

        sample_phys_images.save_to_mrc("output.mrc")

        mock_mrc_new.assert_called_once_with("output.mrc", overwrite=True)
        mock_mrc_file.set_data.assert_called_once()
        assert mock_mrc_file.voxel_size == (1.0, 1.0, 1.0)

    def test_to(self, sample_phys_images):
        device = DEVICES[-1]
        precision = PrecisionLevel.DOUBLE
        
        img = sample_phys_images.to(precision, device)
        
        assert img.images_phys.device == device
        assert img.images_phys.dtype == torch.float64
        assert img.phys_grid.x_pixels.device == device
        assert img.phys_grid.x_pixels.dtype == torch.float64

    def test_normalize_images_max(self, sample_phys_images):
        sample_phys_images.normalize_images(use_max=True)
        mins = torch.amin(sample_phys_images.images_phys, dim=(1, 2))
        maxs = torch.amax(sample_phys_images.images_phys, dim=(1, 2))
        assert torch.allclose(mins, torch.zeros_like(mins))
        assert torch.allclose(maxs, torch.ones_like(maxs))

    def test_normalize_images_lpnorm(self, sample_phys_images):
        original_images = sample_phys_images.images_phys.clone()
        sample_phys_images.normalize_images(ord=2, use_max=False)
        norms = torch.norm(sample_phys_images.images_phys.view(sample_phys_images.n_images, -1), p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms))

    def test_select_images(self, sample_phys_images):
        indices = [1, 3, 5]
        original_images = sample_phys_images.images_phys.clone()
        
        sample_phys_images.select_images(indices)
        
        assert sample_phys_images.n_images == 3
        assert torch.equal(sample_phys_images.images_phys, original_images[indices])

    def test_downsample_images_phys(self, sample_phys_images):
        downsampled = sample_phys_images.downsample_images_phys(downsample_factor=2, type='mean')
        assert downsampled.n_pixels == 64
        assert downsampled.images_phys.shape == (10, 64, 64)
        assert downsampled.box_size == sample_phys_images.box_size

    def test_downsample_images_phys_factor_one(self, sample_phys_images):
        downsampled = sample_phys_images.downsample_images_phys(downsample_factor=1)
        assert downsampled is sample_phys_images

    @patch('cryolike.stack.image.check_nufft_installed')
    def test_transform_to_fourier_cuda_raises_error(self, mock_check_nufft, sample_phys_images, polar_grid_128):
        if torch.cuda.is_available():
            with raises(NotImplementedError, match="CUFINUFFT implementation is currently disabled."):
                sample_phys_images.transform_to_fourier(polar_grid_128, compute_device='cuda')


class TestFourierImages:
    def test_post_init_raises_error_on_shape_mismatch(self, polar_grid_128):
        with raises(ValueError, match="Dimension mismatch"):
            FourierImages(polar_grid=polar_grid_128, box_size=128.0, images_fourier=torch.randn(10, 5, 5))

    def test_to(self, sample_fourier_images):
        device = DEVICES[-1]
        precision = PrecisionLevel.DOUBLE
        
        img = sample_fourier_images.to(get_complex_dtype(precision), device)
        
        assert img.images_fourier.device == device
        assert img.images_fourier.dtype == torch.complex128
        assert img.polar_grid.x_points.device == device
        assert img.polar_grid.x_points.dtype == torch.complex128 # to() converts all tensors

    @patch('cryolike.stack.image.check_nufft_installed')
    def test_transform_to_spatial_cuda_raises_error(self, mock_check_nufft, sample_fourier_images, phys_grid_128):
        if torch.cuda.is_available():
            with raises(NotImplementedError, match="CUFINUFFT implementation is currently disabled."):
                sample_fourier_images.transform_to_spatial(phys_grid_128, compute_device='cuda')


@mark.parametrize("n_pixels", [128, 256])
@mark.parametrize("precision", [PrecisionLevel.SINGLE, PrecisionLevel.DOUBLE])
@mark.parametrize("device", DEVICES)
def test_image_transform_two_way(n_pixels : int, precision: PrecisionLevel, device: torch.device):

    nufft_eps = 1e-6 if precision == PrecisionLevel.SINGLE else 1e-12
    transform_params = {
        "precision" : precision,
        "compute_device" : device,
        "storage_device" : device,
        "eps" : nufft_eps
    }

    box_size = 128.0

    x_1 = 0.1 * box_size
    y_1 = 0.1 * box_size
    x_2 = -0.2 * box_size
    y_2 = -0.2 * box_size

    atom_scale_factor = 4.0
    acc_tol = 1e-9
        
    pixel_size = box_size / n_pixels
    phys_grid = SquaredCartesianGrid2D(n_pixels = n_pixels, box_size = box_size)
    radius_max = np.pi * n_pixels / (2.0 * np.pi) / 2.0
    dist_radii = np.pi / 2.0 / (2.0 * np.pi)
    
    sigma_atom = pixel_size * atom_scale_factor
    sigma_atom_sq = sigma_atom ** 2 

    x_c = phys_grid.x_pixels
    y_c = phys_grid.y_pixels
    Gauss_1_xc = torch.exp(- ((x_c - x_1) ** 2 + (y_c - y_1) ** 2) / (2 * sigma_atom_sq))
    Gauss_1_xc /= (2 * np.pi * sigma_atom_sq)
    Gauss_2_xc = torch.exp(- ((x_c - x_2) ** 2 + (y_c - y_2) ** 2) / (2 * sigma_atom_sq))
    Gauss_2_xc /= (2 * np.pi * sigma_atom_sq)
    image_phys_true = (Gauss_1_xc + Gauss_2_xc) / 2.0 + 0j
    l2norm_image_phys_true = torch.sqrt(torch.sum(torch.abs(image_phys_true) ** 2 * pixel_size ** 2))
    
    n_inplanes = n_pixels * 2
    polar_grid = UniformPolarGrid(
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        quadrature = QuadratureType.GAUSS_JACOBI_BETA_1,
    )
    n_inplanes = polar_grid.n_inplanes

    x_1_scaled = x_1 * 2.0 / box_size * (2.0 * np.pi)
    y_1_scaled = y_1 * 2.0 / box_size * (2.0 * np.pi)
    x_2_scaled = x_2 * 2.0 / box_size * (2.0 * np.pi)
    y_2_scaled = y_2 * 2.0 / box_size * (2.0 * np.pi)

    x_p = polar_grid.x_points
    y_p = polar_grid.y_points

    sigma_atom_sq_scaled = sigma_atom_sq * (2.0 / box_size) ** 2
    Gauss_kp = - (2 * sigma_atom_sq_scaled * (np.pi * polar_grid.radius_shells) ** 2)[:, None]
    T_kp_1 = torch.exp(Gauss_kp - 1j * (x_p * x_1_scaled + y_p * y_1_scaled))
    T_kp_2 = torch.exp(Gauss_kp - 1j * (x_p * x_2_scaled + y_p * y_2_scaled))
    image_fourier_true = (T_kp_1 + T_kp_2) / 2.0
    l2norm_image_fourier_true = torch.sqrt(polar_grid.integrate(torch.abs(image_fourier_true) ** 2))

    for image_repeat in [1, 2]:
        
        images_phys_true_repeat = image_phys_true.unsqueeze(0).repeat(image_repeat, 1, 1)
        images_fourier_true_repeat = image_fourier_true.unsqueeze(0).repeat(image_repeat, 1, 1)
        _phys_images = PhysicalImages(phys_grid, images_phys_true_repeat)
        _fourier_images = FourierImages(polar_grid, box_size, images_fourier_true_repeat)
        
        fourier_images_recover = _phys_images.transform_to_fourier(polar_grid = polar_grid, **transform_params)
        l2norm_image_fourier_recover = torch.sqrt(polar_grid.integrate(torch.abs(fourier_images_recover.images_fourier) ** 2))
        cross_correlation_for = (polar_grid.integrate(images_fourier_true_repeat * torch.conj(fourier_images_recover.images_fourier)) / l2norm_image_fourier_recover / l2norm_image_fourier_true).real
        
        phys_images_recover = _fourier_images.transform_to_spatial(phys_grid = phys_grid, **transform_params)
        images_phys_recover_real = phys_images_recover.images_phys.real
        l2norm_image_phys_real_recover = torch.sqrt(torch.sum(torch.abs(images_phys_recover_real) ** 2 * pixel_size ** 2, dim=(1, 2)))
        cross_correlation_back = (torch.sum(images_phys_true_repeat.real * images_phys_recover_real * pixel_size ** 2, dim=(1,2)) / l2norm_image_phys_real_recover / l2norm_image_phys_true).real

        phys_images_recover_backward = fourier_images_recover.transform_to_spatial(phys_grid = phys_grid, **transform_params)
        images_phys_recover_backward_real = phys_images_recover_backward.images_phys.real
        l2norm_image_phys_real_recover_two = torch.sqrt(torch.sum(torch.abs(images_phys_recover_backward_real) ** 2 * pixel_size ** 2, dim=(1, 2)))
        cross_correlation_two = (torch.sum(images_phys_true_repeat.real * images_phys_recover_backward_real * pixel_size ** 2, dim=(1,2)) / l2norm_image_phys_true / l2norm_image_phys_real_recover_two).real
        
        assert np.allclose(cross_correlation_for, torch.ones_like(cross_correlation_for), atol = acc_tol)
        assert np.allclose(cross_correlation_back, torch.ones_like(cross_correlation_back), atol = acc_tol)
        assert np.allclose(cross_correlation_two, torch.ones_like(cross_correlation_two), atol = acc_tol)
