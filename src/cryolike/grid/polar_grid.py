import torch
import numpy as np
from scipy.special import roots_jacobi
from enum import Enum

from cryolike.util.typechecks import ensure_positive
from cryolike.util.precision import get_float_dtype
from cryolike.util.device_handling import get_device


class QuadratureType(Enum):
    """Quadrature point selection algorithm for polar grids."""
    GAUSS_JACOBI_BETA_1 = 'gauss-jacobi-beta-1'
    GAUSS_JACOBI_BETA_2 = 'gauss-jacobi-beta-2'
    GAUSS_LEGENDRE = 'gauss-legendre'


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
        theta_shell (torch.Tensor): Angles within a shell, shape (n_inplanes,)
        weight_shells (torch.Tensor): Quadrature weights per shell, shape (n_shells,)
        weight_points (torch.Tensor): Quadrature weights per point, shape (n_shells, n_inplanes)
        x_points (torch.Tensor): X-coordinates of each point
        y_points (torch.Tensor): Y-coordinates of each point
    """
    
    radius_max: float
    dist_radii: float
    n_inplanes: int
    quadrature: QuadratureType = QuadratureType.GAUSS_JACOBI_BETA_1

    n_shells: int
    n_points: int
    radius_shells: torch.Tensor
    theta_shell: torch.Tensor
    weight_shells: torch.Tensor
    weight_points: torch.Tensor
    x_points: torch.Tensor
    y_points: torch.Tensor

    def __init__(
        self,
        radius_max: float,
        dist_radii: float,
        n_inplanes: int,
        quadrature: QuadratureType = QuadratureType.GAUSS_JACOBI_BETA_1             
    ):
        # Validate inputs
        ensure_positive(radius_max, "radius_max")
        ensure_positive(dist_radii, "dist_radii")
        ensure_positive(n_inplanes, "n_inplanes")
        self.radius_max = radius_max
        self.dist_radii = dist_radii
        self.n_inplanes = n_inplanes
        
        # Initialize quadrature
        if quadrature == QuadratureType.GAUSS_JACOBI_BETA_1:
            self._gauss_jacobi(beta=1)
        elif quadrature == QuadratureType.GAUSS_JACOBI_BETA_2:
            self._gauss_jacobi(beta=2)
        elif quadrature == QuadratureType.GAUSS_LEGENDRE:
            self._gauss_legendre()
        else:
            raise ValueError(f"Unknown quadrature type: {quadrature}")
        self.quadrature = quadrature
        
        # Set precision and device
        float_dtype = get_float_dtype()
        self.to(dtype=float_dtype, device=torch.device('cpu'))
        
        # Build full grid
        self._build_grid(float_dtype)

    def _build_grid(self, float_dtype: torch.dtype):
        """Construct the full polar grid from radial shells."""
        self.n_points = self.n_shells * self.n_inplanes
        
        # Angular coordinates
        theta_shell = np.linspace(0, 2 * np.pi, self.n_inplanes, endpoint=False)
        self.theta_shell = torch.from_numpy(theta_shell).to(dtype=float_dtype)
        
        # Weights (angular integration factor already included)
        self.weight_points = (self.weight_shells / self.n_inplanes).unsqueeze(1).repeat(1, self.n_inplanes)
        
        # Cartesian coordinates
        self.x_points = self.radius_shells[:,None] * torch.cos(self.theta_shell)[None,:]
        self.y_points = self.radius_shells[:,None] * torch.sin(self.theta_shell)[None,:]

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
        else: # beta == 2
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
                f"quadrature='{self.quadrature.value}', "
                f"n_shells={self.n_shells}, n_points={self.n_points})")

    def to(self, dtype: torch.dtype, device: str | torch.device):
        """Move all tensors to specified dtype and device.
        
        Args:
            dtype: Target data type
            device: Target device
            
        Returns:
            self: Returns self for method chaining
        """
        for attr_name in self.__dict__:
            attr = getattr(self, attr_name, None)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(dtype=dtype, device=device))
        return self
    
    def integrate(self, f: torch.Tensor) -> torch.Tensor:
        """Integrate a function defined on the polar grid using quadrature weights.

        Args:
            f: Function values at each grid point, shape (..., n_shells, n_inplanes)
        Returns:
            Integrated value(s) as a tensor
        """
        if f.ndim < 2:
            raise ValueError(f"Input tensor must have more than 2 dimension, get shape {f.shape}")
        if f.shape[-2] != self.n_shells or f.shape[-1] != self.n_inplanes:
            raise ValueError(f"Input tensor shape {f.shape} does not match grid shape ({self.n_shells}, {self.n_inplanes})")
        return torch.sum(f * self.weight_points.view(*([1] * (f.ndim - 2)), self.n_shells, self.n_inplanes), dim=(-2, -1))
        
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
        device = get_device(device)
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
                x_pts[None, :, :] * x_disp[:, None, None] +
                y_pts[None, :, :] * y_disp[:, None, None]
            )
        )
        return kernel