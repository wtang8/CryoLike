import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from cryolike.grid import UniformPolarGrid

h =  6.62607015e-34 # Planck constant [Js] = [kgm^2/s]
e =  1.60217663e-19 # electron charge [C]; note that [CV] = [J]
c =  2.99792458e8   # speed of light [m/s]
m0 = 9.1093837015e-31 # electron rest mass [kg]

@dataclass
class RelionCTFParameters:

    defocusU: torch.Tensor
    defocusV: torch.Tensor
    defocusAngle: torch.Tensor
    phaseShift: torch.Tensor
    amplitudeContrast: float
    voltage: float
    sphericalAberration: float
    polar_grid: UniformPolarGrid
    box_size: float = 1.0
    anisotropy: bool = True
    cs_corrected: bool = False

    @property
    def n_params(self):
        return len(self.defocusU)

    def __post_init__(self):
        ## check integrity of the CTF parameters
        assert self.defocusU.ndim == 1, "defocusU must be a 1D array"
        assert self.defocusV.ndim == 1, "defocusV must be a 1D array"
        assert self.defocusAngle.ndim == 1, "defocusAngle must be a 1D array"
        assert self.phaseShift.ndim == 1, "phaseShift must be a 1D array"
        assert self.n_params == len(self.defocusV), "defocusU and defocusV must have the same shape"
        assert self.n_params == len(self.defocusAngle), "defocusU and defocusAngle must have the same shape"
        assert self.n_params == len(self.phaseShift), "defocusU and phaseShift must have the same shape"
        assert self.voltage > 0, "voltage must be positive"
        assert self.sphericalAberration >= 0, "sphericalAberration must be non-negative"
        assert self.box_size > 0, "box_size must be positive"
        assert self.amplitudeContrast >= 0, "amplitudeContrast must be non-negative"
        assert self.amplitudeContrast <= 1, "amplitudeContrast must be less than or equal to 1"

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        self.defocusU = self.defocusU.to(dtype=dtype, device=device)
        self.defocusV = self.defocusV.to(dtype=dtype, device=device)
        self.defocusAngle = self.defocusAngle.to(dtype=dtype, device=device)
        self.phaseShift = self.phaseShift.to(dtype=dtype, device=device)
        return self
    

def ctf_relion(params: RelionCTFParameters) -> torch.Tensor:

    defocus = 0.5 * (params.defocusU + params.defocusV)
    astigmatism = 0.5 * torch.abs(params.defocusU - params.defocusV) # abs not necessary
    sphericalAberration = params.sphericalAberration * 1e7 # mm to Angstrom
    voltage = params.voltage * 1e3 # kV to V

    wavelength = h / np.sqrt(2 * m0 * e * voltage) / np.sqrt(1 + e * voltage / (2 * m0 * c * c)) * 1e10 # in Angstrom
    wavelength3 = wavelength ** 3
    coef4 = np.pi / 2 * sphericalAberration * wavelength3
    coef2 = np.pi * wavelength

    r_shell_scaled_ = params.polar_grid.radius_shells * 2.0 / params.box_size
    r_shell_2 = r_shell_scaled_ ** 2
    r_shell_4 = r_shell_2 ** 2
    
    gamma = None
    if params.anisotropy:
        theta_ = params.polar_grid.theta_shell
        local_defocus = defocus[:, None] + astigmatism[:, None] * torch.cos(2 * (theta_[None,:] - params.defocusAngle[:, None]))
        if params.cs_corrected:
            gamma = - coef2 * r_shell_2[None,:,None] * local_defocus[:,None,:] - params.phaseShift[:, None, None]
        else:
            gamma = - coef2 * r_shell_2[None,:,None] * local_defocus[:,None,:] + coef4 * r_shell_4[None,:, None] - params.phaseShift[:, None, None]
    else:
        if params.cs_corrected:
            gamma = - coef2 * r_shell_2[None, :] * defocus[:, None] - params.phaseShift[:, None]
        else:
            gamma = - coef2 * r_shell_2[None, :] * defocus[:, None] + coef4 * r_shell_4[None, :] - params.phaseShift[:, None]
    ctf = - np.sqrt(1 - params.amplitudeContrast * params.amplitudeContrast) * torch.sin(gamma) + params.amplitudeContrast * torch.cos(gamma)
    if not params.anisotropy:
        ctf = ctf.unsqueeze(-1) # (n_params, n_shells, 1)
    return ctf # anisotropic (n_params, n_shells, n_inplanes) or isotropic (n_params, n_shells, 1)

