import numpy as np
import torch
from typing import Union, Optional
from dataclasses import dataclass
from mdtraj import load, Trajectory, Topology

from cryolike.util import ensure_positive

_ATOMIC_RADIUS_RESNAME = {
    "CYS": 2.75, "PHE": 3.20, "LEU": 3.10, "TRP": 3.40,
    "VAL": 2.95, "ILE": 3.10, "MET": 3.10, "HIS": 3.05,
    "TYR": 3.25, "ALA": 2.50, "GLY": 2.25, "PRO": 2.80,
    "ASN": 2.85, "THR": 2.80, "SER": 2.60, "ARG": 3.30,
    "GLN": 3.00, "ASP": 2.80, "LYS": 3.20, "GLU": 2.95,
}

@dataclass
class AtomicModel:
    """Class representing a particle model based on known atomic/protein residue positions.

    Attributes:
        atomic_coordinates (torch.Tensor): Locations of the atoms in the model.
        atom_radii (torch.Tensor): Size of each atom in the model.
        pdb_file (str): The path to the file from which this model was loaded, if any.
            (Otherwise empty string.)
        box_size (float): The side length of the (square) viewing box in which the atoms
            reside.
    """
    coordinates: torch.Tensor
    radii: torch.Tensor
    box_size: float
    file_name: Optional[list[str]] = None

    @property
    def n_frames(self) -> int:
        return self.coordinates.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.coordinates.shape[1]

    def __post_init__(self) -> None:

        if self.coordinates.ndim != 3:
            raise ValueError(f"atomic_coordinates.ndim is expected to be 3, but got {self.coordinates.ndim}")
        ensure_positive(self.box_size, "box size")
        ensure_positive(self.radii, "atomic radii")
        if self.radii.shape != (self.n_atoms,):
            raise ValueError(f"size of radii must match n_atoms in coordinates")
        if self.coordinates.shape != (self.n_frames, self.n_atoms, 3):
            raise ValueError(f"atomic_coordinates.shape is expected to be ({self.n_frames}, {self.n_atoms}, 3), but got {self.coordinates.shape}")
    
    def center_coordinates(self):
        self.coordinates -= torch.mean(self.coordinates, dim = 1, keepdim = True)

    @classmethod
    def read_traj(
        cls,
        box_size: float,
        top_file: str,
        trj_file: Optional[str] = None,
        atom_radii: Optional[torch.Tensor | float] = None,
        atom_selection: Optional[str] = None,
    ) -> 'AtomicModel':
        """Build an atomic model from a PDB file.

        Args:
            pdb_file (str): Path to the PDB file to load
            box_size (float | None, optional): Size of the viewing box. If None (the default),
                a default box size defined in the AtomicModel constructor will be used.
            atom_radii (torch.Tensor | float, optional): Radii of the atoms, either
                as a per-atom array of values or a single value for all atoms. Defaults to 0.1.
            atom_selection (str, optional): Which atoms to choose from the model.
                If using a protein residue model, will be set automatically. Otherwise,
                it should be a valid index of the PDB file's Topology. Defaults to None.
            
        Use protein residue model if atom_radii is None and atom_selection is None, 
            will use the 'name CA' atom selection and read atomic radii from known amino acid sizes.

        Returns:
            AtomicModel: Instantiated atomic model from the PDB file.
        """
        ensure_positive(box_size, "box size")
        use_protein_residue_model = atom_radii is None and atom_selection is None
        if (atom_radii is None) ^ (atom_selection is None):
            raise ValueError(f"atomic_radii is {atom_radii}, but atom_selection is {atom_selection}. Either both should be None or both should be set.")
        if trj_file is None:
            u: Trajectory = load(top_file)
        else:
            u: Trajectory = load(trj_file, top=top_file)
        assert isinstance(u.topology, Topology)
        assert u.xyz is not None
        if u.xyz.ndim == 2:
            u.xyz = u.xyz[None,:,:]
        assert u.xyz.ndim == 3
        pos = torch.tensor(u.xyz[:,:,:], dtype=torch.float32) * 10.0 ## convert from nanometer to Angstrom
        if use_protein_residue_model:
            atom_selection = "name CA"
            res = u.topology.residue
            atom_radii = torch.zeros(u.topology.n_residues, dtype=torch.float32)
            for i in range(u.topology.n_residues):
                resname = res(i).name
                atom_radii[i] = _ATOMIC_RADIUS_RESNAME.get(resname, 3.0)
        if isinstance(atom_radii, float):
            atom_radii = torch.tensor(atom_radii, dtype=torch.float32)
        assert isinstance(atom_radii, torch.Tensor)
        assert atom_selection is not None
        indices = u.topology.select(atom_selection)
        if len(indices) == 0:
            raise ValueError("No atoms selected.")
        if atom_radii.ndim == 0:
            atom_radii = atom_radii.expand(len(indices))
        pos = pos[:,indices,:]
        instance = cls(coordinates=pos, radii=atom_radii, box_size=box_size)
        instance.file_name = [top_file] if trj_file is None else [top_file, trj_file]
        return instance
    
