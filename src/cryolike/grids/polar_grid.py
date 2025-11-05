import torch
import numpy as np
from scipy.special import roots_jacobi
from dataclasses import dataclass, field
from enum import Enum

from cryolike.util.typechecks import ensure_positive
from cryolike.util.precision import get_float_dtype


class QuadratureType(Enum):
    """Quadrature point selection algorithm for polar grids."""
    GAUSS_JACOBI_BETA_1 = 'gauss-jacobi-beta-1'
    GAUSS_JACOBI_BETA_2 = 'gauss-jacobi-beta-2'
    GAUSS_LEGENDRE = 'gauss-legendre'


@dataclass
class UniformPolarGrid:
    """Class implementing uniform polar-coordinate grid.

    Attributes:
        radius_max (float): Maximum radius
        dist_radii (float): Distance between two radial shells
        n_inplanes (int): Number of points in each radial shell
        quadrature (QuadratureType): Quadrature method for radial sampling
        n_shells (int): Number of radial shells
        n_points (int): Total number of points in the grid
        radius_shells (torch.Tensor): Radii of each radial shell, shape (n_shells,)
        radius_points (torch.Tensor): Radii of each point, shape (n_shells * n_inplanes,)
        theta_shell (torch.Tensor): Angles within a shell, shape (n_inplanes,)
        theta_points (torch.Tensor): Angles of each point, shape (n_shells * n_inplanes,)
        weight_shells (torch.Tensor): Quadrature weights per shell, shape (n_shells,)
        weight_points (torch.Tensor): Quadrature weights per point, shape (n_shells * n_inplanes,)
        x_points (torch.Tensor): X-coordinates of each point
        y_points (torch.Tensor): Y-coordinates of each point
    """
    
    radius_max: float
    dist_radii: float
    n_inplanes: int
    quadrature: QuadratureType = QuadratureType.GAUSS_JACOBI_BETA_1
    n_shells: int = field(init=False)
    n_points: int = field(init=False)
    radius_shells: torch.Tensor = field(init=False)
    radius_points: torch.Tensor = field(init=False)
    theta_shell: torch.Tensor = field(init=False)
    theta_points: torch.Tensor = field(init=False)
    weight_shells: torch.Tensor = field(init=False)
    weight_points: torch.Tensor = field(init=False)
    x_points: torch.Tensor = field(init=False)
    y_points: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Validate inputs
        ensure_positive(self.radius_max, "radius_max")
        ensure_positive(self.dist_radii, "dist_radii")
        ensure_positive(self.n_inplanes, "n_inplanes")
        
        # Initialize quadrature
        if self.quadrature == QuadratureType.GAUSS_JACOBI_BETA_1:
            self._gauss_jacobi(beta=1)
        elif self.quadrature == QuadratureType.GAUSS_JACOBI_BETA_2:
            self._gauss_jacobi(beta=2)
        elif self.quadrature == QuadratureType.GAUSS_LEGENDRE:
            self._gauss_legendre()
        else:
            raise ValueError(f"Unknown quadrature type: {self.quadrature}")
        
        # Set precision and device
        float_dtype = get_float_dtype()
        self.to(dtype=float_dtype, device=torch.device('cpu'))
        
        # Build full grid
        self._build_grid(float_dtype)

    def _build_grid(self, float_dtype: torch.dtype):
        """Construct the full polar grid from radial shells."""
        self.n_points = self.n_shells * self.n_inplanes
        
        # Expand radial coordinates
        self.radius_points = self.radius_shells.repeat_interleave(self.n_inplanes)
        
        # Angular coordinates
        theta_shell = np.linspace(0, 2 * np.pi, self.n_inplanes, endpoint=False)
        self.theta_shell = torch.from_numpy(theta_shell).to(dtype=float_dtype)
        self.theta_points = self.theta_shell.repeat(self.n_shells)
        
        # Weights (angular integration factor already included)
        self.weight_points = (self.weight_shells / self.n_inplanes).repeat_interleave(self.n_inplanes)
        
        # Cartesian coordinates
        self.x_points = self.radius_points * torch.cos(self.theta_points)
        self.y_points = self.radius_points * torch.sin(self.theta_points)

    def _gauss_jacobi(self, beta: int):
        """Initialize radial grid using Gauss-Jacobi quadrature.
        
        Args:
            beta (int): Beta parameter for Jacobi polynomial (1 or 2)
        """
        if beta not in [1, 2]:
            raise ValueError('Beta parameter must be 1 or 2 for Gauss-Jacobi quadrature.')
        
        self.n_shells = 1 + int(np.ceil(self.radius_max / self.dist_radii))
        
        # Compute Jacobi quadrature points and weights
        jac_points, jac_weights = roots_jacobi(n=self.n_shells, alpha=0, beta=beta)
        jac_points = torch.from_numpy(jac_points)
        jac_weights = torch.from_numpy(jac_weights)
        
        # Map from [-1, 1] to [0, radius_max]
        self.radius_shells = (jac_points + 1.0) * self.radius_max / 2.0
        
        # Compute integration weights based on beta
        if beta == 1:
            # Weight includes r factor from polar coordinates
            self.weight_shells = jac_weights * (2.0 * np.pi) * (self.radius_max / 2.0) ** 2
        elif beta == 2:
            # Weight includes 1/r factor (for special applications)
            self.weight_shells = jac_weights * (2.0 * np.pi / self.radius_shells) * (self.radius_max / 2.0) ** 3

    def _gauss_legendre(self):
        """Initialize radial grid using Gauss-Legendre quadrature."""
        self.n_shells = 1 + int(np.ceil(self.radius_max / self.dist_radii))
        
        # Compute Legendre quadrature points and weights
        legg_points, legg_weights = np.polynomial.legendre.leggauss(self.n_shells)
        legg_points = torch.from_numpy(legg_points)
        legg_weights = torch.from_numpy(legg_weights)
        
        # Map from [-1, 1] to [0, radius_max]
        self.radius_shells = (legg_points + 1.0) * self.radius_max / 2.0
        
        # Weight includes r factor and normalization
        self.weight_shells = legg_weights * self.radius_shells * self.radius_max * np.pi

    def __repr__(self) -> str:
        return (f"UniformPolarGrid(radius_max={self.radius_max}, "
                f"dist_radii={self.dist_radii}, n_inplanes={self.n_inplanes}, "
                f"quadrature={self.quadrature.value}, "
                f"n_shells={self.n_shells}, n_points={self.n_points})")

    def to(self, dtype: torch.dtype, device: str | torch.device):
        """Move all tensors to specified dtype and device.
        
        Args:
            dtype: Target data type
            device: Target device
            
        Returns:
            self: Returns self for method chaining
        """
        if hasattr(self, 'radius_shells'):
            self.radius_shells = self.radius_shells.to(dtype=dtype, device=device)
            self.weight_shells = self.weight_shells.to(dtype=dtype, device=device)
        
        if hasattr(self, 'radius_points'):
            self.radius_points = self.radius_points.to(dtype=dtype, device=device)
            self.theta_shell = self.theta_shell.to(dtype=dtype, device=device)
            self.theta_points = self.theta_points.to(dtype=dtype, device=device)
            self.weight_points = self.weight_points.to(dtype=dtype, device=device)
            self.x_points = self.x_points.to(dtype=dtype, device=device)
            self.y_points = self.y_points.to(dtype=dtype, device=device)
        
        return self

    def get_fourier_translation_kernel(
        self,
        x_displacements_angstrom: torch.Tensor,
        y_displacements_angstrom: torch.Tensor,
        box_size_x: float = 2.0,
        box_size_y: float = 2.0,
        device: str | torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Get a Fourier-space translation kernel.
        
        Computes a kernel that can be pointwise-multiplied with a Fourier-space 
        image to accomplish translation(s) in physical space.

        Args:
            x_displacements_angstrom: X-displacement(s) to apply in Angstrom
            y_displacements_angstrom: Y-displacement(s) to apply in Angstrom
            box_size_x: Total width of viewing box in Angstrom (default: 2.0)
            box_size_y: Total height of viewing box in Angstrom (default: 2.0)
            device: Device to place resulting tensor on (default: CPU)
            
        Returns:
            torch.Tensor: Translation kernel of shape 
                [n_displacements, n_shells, n_inplanes]
        """
        device = device if torch.cuda.is_available() else torch.device('cpu')
        float_dtype = get_float_dtype()
        
        # Convert displacements from Angstrom to normalized coordinates [-1, 1]
        x_disp = x_displacements_angstrom.to(dtype=float_dtype, device=device) * 2.0 / box_size_x
        y_disp = y_displacements_angstrom.to(dtype=float_dtype, device=device) * 2.0 / box_size_y
        
        # Get grid points on target device
        x_pts = self.x_points.to(dtype=float_dtype, device=device)
        y_pts = self.y_points.to(dtype=float_dtype, device=device)
        
        # Compute phase shift: exp(-2πi * (k · displacement))
        kernel = torch.exp(
            -2.0 * np.pi * 1j * (
                x_pts[None, :] * x_disp[:, None] +
                y_pts[None, :] * y_disp[:, None]
            )
        )
        
        # Reshape to [n_displacements, n_shells, n_inplanes]
        return kernel.reshape(-1, self.n_shells, self.n_inplanes)