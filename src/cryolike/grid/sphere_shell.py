from typing import Optional, NamedTuple
import numpy as np
import torch
from scipy.special import roots_legendre, roots_jacobi
from math import ceil

from cryolike.util import ensure_positive


class SphereShell:
    """Class to sample points on a sphere shell of known radius r.

    Attributes:
        radius (float): Radius of the sphere in Angstrom. Positive finite.
        dist_eq (float): Distance between points at the equator. Positive finite.
        azimuthal_sampling (SamplingStrategy): Whether the grid is uniform or adaptive.
        n_points (int): number of points
        n_polar_circles (int): number of polar circles
        n_azimus_each_circle (np.ndarray): number of azimuthal points at each polar circle
        polar_circles (torch.Tensor): polar angles at each polar circle
        weights_polar (torch.Tensor): weight at each point
        azimu_points (torch.Tensor): azimuthal angles at each point
        polar_points (torch.Tensor): polar angles at each point
        weight_points (torch.Tensor): weight at each point
        cartesian_points (Optional[CartesianShell]): If set, a NamedTuple of four float arrays
            (x_points, y_points, z_points, xyz_points) defining cartesian coordinates of each
            point.
    """
    radius: float
    dist_eq: float
    n_points : int
    n_polar_circles : int
    n_azimus_each_circle: np.ndarray
    polar_circles: torch.Tensor
    weights_polar: torch.Tensor
    azimu_points: torch.Tensor
    polar_points: torch.Tensor
    weight_points: torch.Tensor
    xyz_points: Optional[torch.Tensor] = None

    def __init__(self,
        radius: float = 1.0,
        dist_eq: float = 1.0 / (2.0 * np.pi),
        uniform_azimuthal_sampling : bool = True,
        store_cartesian_points : bool = False,
    ) -> None:
        """Class to sample points on a sphere shell of known radius r.

        Args:
            radius (float, optional): Radius of the sphere in Angstrom. Must be positive finite. Defaults to 1.0.
            dist_eq (float, optional): Distance between points at equator. Must be positive finite.
                Defaults to 1.0/(2.0 * np.pi).
            azimuthal_sampling (SamplingStrategy, optional): Enum indicating whether to use uniform sampling
                (the default) or adaptive sampling.
            compute_cartesian (bool, optional): Whether to populate a cartesian grid for the sphere. If True,
                the resulting shell will have a cartesian_points member set. Defaults to True.
        """
        ensure_positive(radius, "spherical shell radius")
        ensure_positive(dist_eq, "spherical shell equatorial point distance dist_eq")
        self.radius = radius
        self.dist_eq = dist_eq
        angular_frequency = 2. * np.pi * self.radius / self.dist_eq
        n_point_eq = 3 + int(round(angular_frequency))  # number of points on equator
        self.n_polar_circles = 3 + ceil(n_point_eq / 2)
        lgnd_nodes, lgnd_weights = roots_legendre(self.n_polar_circles)
        polar_circles = np.arccos(lgnd_nodes)   # polar angle of each polar circle
        self.weights_polar = torch.from_numpy(lgnd_weights)
        n_azimu_max = 3 + int(round(angular_frequency))
        if uniform_azimuthal_sampling:  # uniform sampling
            self.n_azimus_each_circle = np.ones(self.n_polar_circles, dtype=int) * n_azimu_max
        else:  # adaptive sampling
            sin_polar_circles = np.sin(polar_circles)
            self.n_azimus_each_circle = 3 + np.round(angular_frequency * sin_polar_circles).astype(int)
        self.polar_circles = torch.from_numpy(polar_circles)
        self.n_points = int(np.sum(self.n_azimus_each_circle))
        self.azimu_points = torch.zeros(self.n_points, dtype = torch.float64)
        self.polar_points = torch.zeros(self.n_points, dtype = torch.float64)
        self.weight_points = torch.zeros(self.n_points, dtype = torch.float64)
        radius_sq = self.radius ** 2
        i_point : int = 0
        for i, n_azimus in enumerate(self.n_azimus_each_circle):
            azimu_per_polar = torch.tensor(np.linspace(0, 2 * np.pi, n_azimus, endpoint = False), dtype=torch.float64)
            d_azimu = azimu_per_polar[1] - azimu_per_polar[0]
            self.azimu_points[i_point : i_point + n_azimus] = azimu_per_polar
            self.polar_points[i_point : i_point + n_azimus] = self.polar_circles[i]
            self.weight_points[i_point : i_point + n_azimus] = self.weights_polar[i] * radius_sq * d_azimu  ## integrate to 4pir^2
            i_point += n_azimus
        assert i_point == self.n_points
        if store_cartesian_points:
            self.xyz_points = self.cartesian_points()
        
    def cartesian_points(self):

        cos_azimu_ = torch.cos(self.azimu_points)
        sin_azimu_ = torch.sin(self.azimu_points)
        cos_polar_ = torch.cos(self.polar_points)
        sin_polar_ = torch.sin(self.polar_points)

        x_points = self.radius * cos_azimu_ * sin_polar_
        y_points = self.radius * sin_azimu_ * sin_polar_
        z_points = self.radius * cos_polar_
        xyz_points = torch.stack((x_points, y_points, z_points), dim = 1)

        return xyz_points

    def integrate(self, f: torch.Tensor):            
        # integrate function f over the sphere shell
        if not isinstance(f, torch.Tensor):
            raise TypeError("f must be a torch.Tensor")
        if f.shape[-1] != self.n_points:
            raise ValueError(' %% error: f.shape[0] != self.n_points')
        return f @ self.weight_points.to(dtype = f.real.dtype, device = f.device)
