import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from typing import Union, Optional
from dataclasses import dataclass

from mdtraj import Topology
from cryolike.model.atomic import AtomicModel, _ATOMIC_RADIUS_RESNAME


class TestAtomicModel(unittest.TestCase):
    """
    Unit tests for the AtomicModel class, primarily focusing on the read_traj classmethod.
    We use patching to mock the external 'mdtraj' library.
    """

    @patch('cryolike.model.atomic.load')
    def test_read_traj_protein_model_full(self, mock_load):
        """Tests default behavior: CA atom selection, radius lookup, and two files (top + trj)."""
        
        # --- Setup Mock Data ---
        N_RES = 3 # Number of residues (and CA atoms selected)
        N_TOTAL_ATOMS = 10 # Total atoms in the mock trajectory
        
        # Mock Coordinates (2 frames, 10 atoms, in nm)
        mock_xyz = np.random.rand(2, N_TOTAL_ATOMS, 3).astype(np.float32)
        
        # Define the residue sequence names and the expected atom_radii
        RESIDUE_NAMES = ["CYS", "PHE", "ALA"]
        expected_radii = torch.tensor([
            _ATOMIC_RADIUS_RESNAME["CYS"], 
            _ATOMIC_RADIUS_RESNAME["PHE"], 
            _ATOMIC_RADIUS_RESNAME["ALA"], 
        ], dtype=torch.float32)

        # Mock the u.topology.residue(i) callable
        def mock_residue_getter(i):
            mock_res = MagicMock()
            mock_res.name = RESIDUE_NAMES[i]
            return mock_res
        
        # Define the indices that mdtraj.select("name CA") will return
        CA_INDICES = np.array([1, 4, 7]) 
        
        # --- Configure Mocks ---
        mock_topology = MagicMock(spec=Topology, n_residues=N_RES, residue=mock_residue_getter)
        mock_topology.select.return_value = CA_INDICES
        
        mock_trajectory = MagicMock(xyz=mock_xyz, topology=mock_topology)
        mock_load.return_value = mock_trajectory

        BOX_SIZE = 50.0
        TOP_FILE = "mock.pdb"
        TRJ_FILE = "mock.xtc"
        
        # --- Call Function ---
        model = AtomicModel.read_traj(
            box_size=BOX_SIZE,
            top_file=TOP_FILE,
            trj_file=TRJ_FILE,
            use_protein_residue_model=True,
        )
        
        # --- Assertions ---
        
        # 1. mdtraj.load was called correctly
        mock_load.assert_called_once_with(TRJ_FILE, top=TOP_FILE)
        
        # 2. Correct atom selection was performed for protein model
        mock_topology.select.assert_called_once_with("name CA")
        
        # 3. Model dimensions and metadata
        self.assertEqual(model.n_frames, 2)
        self.assertEqual(model.n_atoms, N_RES) # Only CA atoms are selected
        self.assertEqual(model.box_size, BOX_SIZE)
        self.assertEqual(model.file_name, [TOP_FILE, TRJ_FILE])
        
        # 4. Coordinates check (nm -> Angstrom conversion)
        expected_coordinates = torch.tensor(mock_xyz[:, CA_INDICES, :] * 10.0, dtype=torch.float32)
        self.assertTrue(torch.equal(model.coordinates, expected_coordinates))
        
        # 5. atom_radii check (lookup logic)
        self.assertTrue(torch.equal(model.atom_radii, expected_radii))

    @patch('cryolike.model.atomic.load')
    def test_read_traj_explicit_single_file_and_float_radius(self, mock_load):
        """Tests explicit atom selection and single float radius, loading only a PDB file."""
        
        # --- Setup Mock Data ---
        N_TOTAL_ATOMS = 5 
        SELECTED_COUNT = 3
        
        # Single frame (2D array, which the function should handle by expanding to 3D)
        mock_xyz = np.random.rand(N_TOTAL_ATOMS, 3).astype(np.float32) 
        
        ATOM_SELECTION = "resname HOH"
        SELECTED_INDICES = np.array([0, 2, 4]) 
        EXPLICIT_RADIUS = 1.5 
        
        # --- Configure Mocks ---
        mock_topology = MagicMock(spec=Topology)
        mock_topology.select.return_value = SELECTED_INDICES
        
        mock_trajectory = MagicMock(xyz=mock_xyz, topology=mock_topology)
        mock_load.return_value = mock_trajectory

        BOX_SIZE = 100.0
        TOP_FILE = "water.pdb" # Used as the only file
        
        # --- Call Function ---
        model = AtomicModel.read_traj(
            box_size=BOX_SIZE,
            top_file=TOP_FILE,
            trj_file=None, # Test single file load
            atom_radii=EXPLICIT_RADIUS,
            atom_selection=ATOM_SELECTION,
        )
        
        # --- Assertions ---
        
        # 1. mdtraj.load was called correctly (only top_file)
        mock_load.assert_called_once_with(TOP_FILE) 
        
        # 2. Correct atom selection was performed
        mock_topology.select.assert_called_once_with(ATOM_SELECTION)
        
        # 3. Model dimensions and metadata
        self.assertEqual(model.n_frames, 1) # Should be 1 after expansion from 2D
        self.assertEqual(model.n_atoms, SELECTED_COUNT) 
        self.assertEqual(model.file_name, [TOP_FILE])

        # 4. atom_radii check (float conversion to tensor)
        expected_radii = torch.full((SELECTED_COUNT,), EXPLICIT_RADIUS, dtype=torch.float32)
        self.assertTrue(torch.equal(model.atom_radii, expected_radii))

        # 5. Coordinates check (should handle 2D to 3D expansion implicitly)
        expected_coordinates = torch.tensor(mock_xyz[SELECTED_INDICES, :] * 10.0, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.equal(model.coordinates, expected_coordinates))

    @patch('cryolike.model.atomic.load')
    def test_read_traj_explicit_tensor_radius(self, mock_load):
        """Tests explicit atom selection and a pre-defined per-atom tensor radius."""
        
        # --- Setup Mock Data ---
        N_TOTAL_ATOMS = 5
        SELECTED_COUNT = 3
        
        mock_xyz = np.random.rand(2, N_TOTAL_ATOMS, 3).astype(np.float32)
        
        ATOM_SELECTION = "ions"
        SELECTED_INDICES = np.array([0, 2, 4]) 
        EXPLICIT_RADII = torch.tensor([1.1, 1.2, 1.3], dtype=torch.float32) 
        
        # --- Configure Mocks ---
        mock_topology = MagicMock(spec=Topology)
        mock_topology.select.return_value = SELECTED_INDICES
        mock_trajectory = MagicMock(xyz=mock_xyz, topology=mock_topology)
        mock_load.return_value = mock_trajectory

        BOX_SIZE = 75.0
        TOP_FILE = "ions.top"
        TRJ_FILE = "ions.trj"
        
        # --- Call Function ---
        model = AtomicModel.read_traj(
            box_size=BOX_SIZE,
            top_file=TOP_FILE,
            trj_file=TRJ_FILE,
            atom_radii=EXPLICIT_RADII,
            atom_selection=ATOM_SELECTION,
        )
        
        # --- Assertions ---
        self.assertEqual(model.n_frames, 2)
        self.assertEqual(model.n_atoms, SELECTED_COUNT) 
        self.assertTrue(torch.equal(model.atom_radii, EXPLICIT_RADII))

    def test_read_traj_error_mismatched_inputs(self):
        """Tests error when only one of atom_radii or atom_selection is provided."""
        
        BOX_SIZE = 10.0
        TOP_FILE = "mock.pdb"

        # Case 1: atom_radii set, selection is None, and not using protein model
        with self.assertRaisesRegex(ValueError, "Both should be set if not using protein residue model."):
            AtomicModel.read_traj(
                box_size=BOX_SIZE,
                top_file=TOP_FILE,
                use_protein_residue_model=False,
                atom_radii=0.5,
                atom_selection=None,
            )
        
        # Case 2: atom_radii is None, selection set, and not using protein model
        with self.assertRaisesRegex(ValueError, "Both should be set if not using protein residue model."):
            AtomicModel.read_traj(
                box_size=BOX_SIZE,
                top_file=TOP_FILE,
                use_protein_residue_model=False,
                atom_radii=None,
                atom_selection="all",
            )
            
    @patch('cryolike.model.atomic.load')
    def test_read_traj_error_no_atoms_selected(self, mock_load):
        """Tests error when atom selection returns an empty list."""
        
        # --- Configure Mocks ---
        mock_topology = MagicMock(spec=Topology)
        mock_topology.select.return_value = np.array([]) # No atoms selected
        
        mock_trajectory = MagicMock(xyz=np.random.rand(1, 5, 3).astype(np.float32), topology=mock_topology)
        mock_load.return_value = mock_trajectory

        BOX_SIZE = 10.0
        TOP_FILE = "mock.pdb"
        
        # --- Call Function (Explicit non-protein model to force selection) ---
        with self.assertRaisesRegex(ValueError, "No atoms selected"):
            AtomicModel.read_traj(
                box_size=BOX_SIZE,
                top_file=TOP_FILE,
                atom_radii=1.0,
                atom_selection="nothing",
            )

    @patch('cryolike.model.atomic.load')
    def test_read_traj_error_atom_radii_shape_error(self, mock_load):
        """Tests error when atom atom_radii tensor does not match number of selected atoms."""

        mock_topology = MagicMock(spec=Topology)
        mock_topology.select.return_value = np.array([0, 1, 2])
        mock_trajectory = MagicMock(xyz=np.random.rand(1, 3, 3).astype(np.float32), topology=mock_topology)
        mock_load.return_value = mock_trajectory

        BOX_SIZE = 10.0
        TOP_FILE = "mock.pdb"
        
        # --- Call Function ---
        with self.assertRaisesRegex(ValueError, "size of atom_radii must match number of selected atoms"):
            AtomicModel.read_traj(
                box_size=BOX_SIZE,
                top_file=TOP_FILE,
                atom_radii=torch.rand(4),
                atom_selection="all",
            )


class TestAtomicModelPostInit(unittest.TestCase):
    """Tests for the AtomicModel constructor and post-init validation."""

    def test_post_init_ndim_error(self):
        """Tests ValueError if coordinates tensor is not 3-dimensional."""
        with self.assertRaisesRegex(ValueError, "atomic_coordinates.ndim is expected to be 3, but got 2"):
            AtomicModel(
                coordinates=torch.rand(5, 3), # 2D tensor
                atom_radii=torch.rand(5),
                box_size=10.0
            )

    def test_post_init_shape_error(self):
        """Tests ValueError if coordinates tensor's last dimension is not 3."""
        with self.assertRaisesRegex(ValueError, r"atomic_coordinates.shape is expected to be \(1, 5, 3\), but got torch.Size\(\[1, 5, 4\]\)"):
            AtomicModel(
                coordinates=torch.rand(1, 5, 4), # Last dimension is 4
                atom_radii=torch.rand(5),
                box_size=10.0
            )

    def test_atom_radii_shape_error(self):
        """Tests ValueError if atom_radii tensor does not match n_atoms."""
        with self.assertRaisesRegex(ValueError, "size of atom_radii must match n_atoms in coordinates"):
            AtomicModel(
                coordinates=torch.rand(1, 5, 3),
                atom_radii=torch.rand(4),
                box_size=10.0
            )

    def test_center_coordinates(self):
        """Tests the center_coordinates method."""
        # Create coordinates that are not centered at the origin
        coords = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32)
        model = AtomicModel(
            coordinates=coords,
            atom_radii=torch.rand(2),
            box_size=100.0
        )
        
        model.center_coordinates()
        
        # Check that the mean of coordinates along the atom dimension is close to zero
        mean_coords = torch.mean(model.coordinates, dim=1)
        self.assertTrue(torch.allclose(mean_coords, torch.zeros_like(mean_coords), atol=1e-6))