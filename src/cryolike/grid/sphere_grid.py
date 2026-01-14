from typing import Optional, NamedTuple
import numpy as np
import torch
from scipy.special import roots_legendre, roots_jacobi
from math import ceil
from copy import deepcopy

from .sphere_shell import SphereShell
from cryolike.util import ensure_positive


class SphereGrid:
    """Class to sample points on a spherical grid comprised of shells.

    Attributes:
        radius_max (float): Maximum radius of the sphere
        dist_eq (float): minimum distance between two points at the equator
        equal_shell (bool): if True, all shells have the same angular points
        n_shells (int): number of radial shells
        radius_shells (torch.Tensor): radius of each radial shell
        weights_radius_shells (torch.Tensor): weight of each radial shell
        n_points (int): total number of points
        radius_points (torch.Tensor): radius of each point
        polar_points (torch.Tensor): polar angle of each point
        azimu_points (torch.Tensor): azimuthal angle of each point
        weight_points (torch.Tensor): weight of each point
        shells (list[SphereShell]): list of radial shells
        point_shell_start_indices (IntArrayType): start index of each shell within the point list
        cartesian_points (Optional[CartesianShell]): If set, a NamedTuple of four float arrays
            (x_points, y_points, z_points, xyz_points) defining cartesian coordinates of each
            point
    """
    radius_max: float
    dist_eq: float
    uniform_azimuthal_sampling: bool
    equal_shell: bool
    uniform_dist: bool
    
    n_shells: int
    radius_shells: torch.Tensor
    weights_radius_shells: torch.Tensor
    n_points: int
    radius_points: torch.Tensor
    polar_points: torch.Tensor
    azimu_points: torch.Tensor
    weight_points: torch.Tensor
    xyz_points: torch.Tensor
    point_shell_start_indices: np.ndarray
    shells: list[SphereShell]

    def __init__(self,
        radius_max: float = 1.0,
        dist_eq: float = 1.0 / (2.0 * np.pi),
        uniform_azimuthal_sampling: bool = True,
        equal_shell: bool = True,
        uniform_dist: bool = True,
    ) -> None:
        """Class to sample points on a spherical grid comprised of shells.

        Args:
            radius_max (float, optional): maximum radius of the sphere. Defaults to 1.0.
            dist_eq (float, optional): minimum distance between two points at the equator. Defaults to 1.0/(2.0 * np.pi).
            azimuthal_sampling (bool): 'uniform' (true) or 'adaptive' azimuthal sampling for various polar circles
            equal_shell (bool): if True, all shells have the same angular points. Defaults to False.
            uniform_dist (bool, optional): 'uniform' (true) or 'adaptive' (false) distance between points for difference shells.
        """
        ensure_positive(radius_max, "maximum spherical radius")
        ensure_positive(dist_eq, "minimum equatorial distance")
        self.radius_max = radius_max
        self.dist_eq = dist_eq
        self.uniform_azimuthal_sampling = uniform_azimuthal_sampling
        self.equal_shell = equal_shell
        self.uniform_dist = uniform_dist
        self._sample_shell_radii()
        if equal_shell:
            self._build_grid_equal_shells()
        else:
            self._build_grid_varied_shells(uniform_dist)

    def _sample_shell_radii(self):
        ## sample shell radii
        self.n_shells = 1 + int(np.ceil(self.radius_max / self.dist_eq))          # number of radial shells
        points_jacobi, weights_jacobi = roots_jacobi(self.n_shells, 0, 2)         # Gauss-Jacobi quadrature.
        self.radius_shells = torch.tensor(
            (points_jacobi + 1.0) * self.radius_max / 2
        )  # radius of each radial shell
        self.weights_radius_shells = torch.tensor(
            weights_jacobi * (self.radius_max / 2) ** 3
        )  # weight of each radial shell

    def _build_grid_equal_shells(self):
        shell = SphereShell(
            radius = 1.0, 
            dist_eq = self.dist_eq / self.radius_max,
            uniform_azimuthal_sampling = self.uniform_azimuthal_sampling
        )
        self.weight_points = torch.tensor([], dtype=torch.float64)
        self.shells = []
        for i_s in range(self.n_shells):
            this_shell = deepcopy(shell)
            this_shell.radius = self.radius_shells[i_s].item()
            this_shell.weight_points *= self.radius_shells[i_s] ** 2
            this_shell.xyz_points *= this_shell.radius
            self.weight_points = torch.concatenate((self.weight_points, this_shell.weight_points * self.weights_radius_shells[i_s] / self.radius_shells[i_s] ** 2))
            self.shells.append(this_shell)
        self.n_points = self.n_shells * shell.n_points
        self.radius_points = self.radius_shells.repeat_interleave(shell.n_points)
        self.polar_points = shell.polar_points.repeat(self.n_shells)
        self.azimu_points = shell.azimu_points.repeat(self.n_shells)
        self.point_shell_start_indices = np.arange(self.n_shells + 1) * shell.n_points

    def _build_grid_varied_shells(self, uniform_dist: bool = False):
        self.n_points = 0
        if uniform_dist:
            dist_eq_shells = self.dist_eq * torch.ones(self.n_shells, dtype=torch.float64)
        else:
            dist_eq_shells = self.dist_eq * self.radius_shells / self.radius_max
        
        self.point_shell_start_indices = np.zeros(self.n_shells + 1, dtype=int)
        self.shells = []
        self.radius_points = torch.tensor([], dtype=torch.float64)
        self.polar_points = torch.tensor([], dtype=torch.float64)
        self.azimu_points = torch.tensor([], dtype=torch.float64)
        self.weight_points = torch.tensor([], dtype=torch.float64)
        for i_s in range(self.n_shells):
            this_shell = SphereShell(
                radius = self.radius_shells[i_s].item(), 
                dist_eq = dist_eq_shells[i_s].item(), # / self.radius_max,
                uniform_azimuthal_sampling = self.uniform_azimuthal_sampling
            )
            print("radius:", this_shell.radius, "dist_eq:", this_shell.dist_eq)
            self.shells.append(this_shell)
            self.point_shell_start_indices[i_s + 1] = self.point_shell_start_indices[i_s] + this_shell.n_points
            self.radius_points = torch.concatenate((self.radius_points, this_shell.radius * torch.ones(this_shell.n_points, dtype=torch.float64)))
            self.polar_points = torch.concatenate((self.polar_points, this_shell.polar_points))
            self.azimu_points = torch.concatenate((self.azimu_points, this_shell.azimu_points))
            self.weight_points = torch.concatenate((self.weight_points, this_shell.weight_points * self.weights_radius_shells[i_s] / self.radius_shells[i_s] ** 2))
        self.n_points = self.point_shell_start_indices[-1]

    def integrate(self,
        f: torch.Tensor, # functional values to integrate
        use_riesz_integration: bool = False
    ) -> torch.Tensor:
        # integrate function f over the sphere shell
        if f.shape[-1] != self.n_points:
            raise ValueError(' %% error: f.shape[-1] != self.n_points')
        assert isinstance(f, torch.Tensor)
        weights = self.weight_points.to(dtype = f.dtype, device = f.device)
        if use_riesz_integration:
            weights /= self.radius_points
        return f @ weights
