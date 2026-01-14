import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from cryolike.stack.template import (
    _fourier_circles,
    Templates,
    AtomShape,
)
from cryolike.model import AtomicModel
from cryolike.pose import ViewingAngles
from cryolike.grid import UniformPolarGrid
from cryolike.util import PrecisionLevel


class TestTemplateHelpers(unittest.TestCase):
    def test_fourier_circles(self):
        """Tests the _fourier_circles helper function."""
        n_angles = 2
        n_thetas = 5
        thetas = torch.linspace(0, 2 * np.pi, n_thetas)
        azimus = torch.tensor([0.0])
        polars = torch.tensor([0.0])
        gammas = torch.zeros(n_angles)

        xyz = _fourier_circles(thetas, polars, azimus, gammas)

        # 1. Check output shape
        self.assertEqual(xyz.shape, (n_angles, n_thetas, 3))

        # 2. Check that points lie on a unit sphere by checking their norm
        norms = torch.linalg.norm(xyz, dim=2)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))

        # 3. Check a known case (polar angle = 0, azimu = 0)
        # Should trace a circle in the xy-plane
        self.assertTrue(torch.allclose(xyz[0, :, 0], torch.cos(thetas), atol=1e-7))
        self.assertTrue(torch.allclose(xyz[0, :, 1], torch.sin(thetas), atol=1e-7))
        self.assertTrue(torch.allclose(xyz[0, :, 2], torch.zeros_like(thetas), atol=1e-7))


class TestTemplates(unittest.TestCase):
    def setUp(self):
        """Set up common mock objects for Templates tests."""
        self.n_templates_per_frame = 5
        self.n_frames = 2
        self.n_images = self.n_frames * self.n_templates_per_frame
        self.n_shells = 10
        self.n_inplanes = 20

        self.mock_viewing_angles = MagicMock(spec=ViewingAngles)
        self.mock_viewing_angles.n_angles = self.n_templates_per_frame

        self.mock_polar_grid = MagicMock(spec=UniformPolarGrid)
        self.mock_polar_grid.n_shells = self.n_shells
        self.mock_polar_grid.n_inplanes = self.n_inplanes

        self.images_fourier = torch.rand(self.n_images, self.n_shells, self.n_inplanes, dtype=torch.complex64)
        self.box_size = 100.0

    def test_init_success(self):
        """Test successful initialization of Templates."""
        templates = Templates(
            viewing_angles=self.mock_viewing_angles,
            images_fourier=self.images_fourier,
            polar_grid=self.mock_polar_grid,
            box_size=self.box_size,
            n_frames=self.n_frames
        )
        self.assertIsInstance(templates, Templates)
        self.assertEqual(templates.n_images, self.n_images)
        self.assertEqual(templates.n_frames, self.n_frames)

    def test_init_viewing_angle_mismatch(self):
        """Test ValueError when number of viewing angles does not match number of templates."""
        mock_va_wrong_count = MagicMock(spec=ViewingAngles)
        mock_va_wrong_count.n_angles = self.n_templates_per_frame + 1  # Mismatch

        with self.assertRaisesRegex(ValueError, "Number of viewing angles must match number of templates."):
            Templates(
                viewing_angles=mock_va_wrong_count,
                images_fourier=self.images_fourier,
                polar_grid=self.mock_polar_grid,
                box_size=self.box_size,
                n_frames=self.n_frames
            )

    @patch('cryolike.stack.template.make_uniform_hard_sphere')
    @patch('cryolike.stack.template.make_uniform_gaussian')
    def test_generate_from_positions(self, mock_make_gaussian, mock_make_hard_sphere):
        """Test the generate_from_positions class method."""
        # --- Setup Mocks ---
        # The generator functions will produce a 4D tensor
        generated_fourier_images = torch.rand(self.n_frames, self.n_templates_per_frame, self.n_shells, self.n_inplanes, dtype=torch.complex64)
        mock_make_gaussian.return_value = generated_fourier_images
        mock_make_hard_sphere.return_value = generated_fourier_images

        mock_atomic_model = MagicMock(spec=AtomicModel)
        mock_atomic_model.use_protein_residue_model = False
        mock_atomic_model.n_frames = self.n_frames
        mock_atomic_model.to.return_value = mock_atomic_model # Ensure .to() returns self

        self.mock_viewing_angles.to.return_value = self.mock_viewing_angles
        self.mock_polar_grid.to.return_value = self.mock_polar_grid

        # --- Test Gaussian ---
        templates_gauss = Templates.generate_from_positions(
            atomic_model=mock_atomic_model,
            viewing_angles=self.mock_viewing_angles,
            polar_grid=self.mock_polar_grid,
            box_size=self.box_size,
            atom_shape=AtomShape.GAUSSIAN
        )
        mock_make_gaussian.assert_called_once()
        mock_make_hard_sphere.assert_not_called()
        
        # The final images_fourier should be reshaped to (n_frames * n_templates, ...)
        expected_shape = (self.n_frames * self.n_templates_per_frame, self.n_shells, self.n_inplanes)
        self.assertEqual(templates_gauss.images_fourier.shape, expected_shape)
        self.assertEqual(templates_gauss.n_images, self.n_images)
        self.assertEqual(templates_gauss.n_frames, self.n_frames)

        # --- Test Hard Sphere ---
        mock_make_gaussian.reset_mock()
        mock_make_hard_sphere.reset_mock()
        
        templates_hs = Templates.generate_from_positions(
            atomic_model=mock_atomic_model,
            viewing_angles=self.mock_viewing_angles,
            polar_grid=self.mock_polar_grid,
            box_size=self.box_size,
            atom_shape=AtomShape.HARD_SPHERE
        )
        mock_make_gaussian.assert_not_called()
        mock_make_hard_sphere.assert_called_once()

        # --- Test Default for Protein Model ---
        mock_make_gaussian.reset_mock()
        mock_make_hard_sphere.reset_mock()
        mock_atomic_model.use_protein_residue_model = True
        
        Templates.generate_from_positions(
            atomic_model=mock_atomic_model,
            viewing_angles=self.mock_viewing_angles,
            polar_grid=self.mock_polar_grid,
            box_size=self.box_size,
            atom_shape=AtomShape.DEFAULT # Let it use the default logic
        )
        # Should default to hard sphere when using protein model
        mock_make_gaussian.assert_not_called()
        mock_make_hard_sphere.assert_called_once()

        # --- Test Default for non-protein model ---
        mock_make_gaussian.reset_mock()
        mock_make_hard_sphere.reset_mock()
        mock_atomic_model.use_protein_residue_model = False

        Templates.generate_from_positions(
            atomic_model=mock_atomic_model,
            viewing_angles=self.mock_viewing_angles,
            polar_grid=self.mock_polar_grid,
            box_size=self.box_size,
            atom_shape=AtomShape.DEFAULT # Let it use the default logic
        )
        # Should default to hard sphere when shape is DEFAULT
        mock_make_gaussian.assert_not_called()
        mock_make_hard_sphere.assert_called_once()


if __name__ == '__main__':
    unittest.main()