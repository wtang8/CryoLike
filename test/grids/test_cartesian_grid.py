import pytest
import torch
import numpy as np
from pytest import mark
from cryolike.grids.cartesian_grid import SquaredCartesianGrid2D

from cryolike.util.precision import set_precision

set_precision('single')

def test_grid_initialization():
    # Test basic initialization
    grid = SquaredCartesianGrid2D(n_pixels=32, box_size=32.0, endpoint=False)
    assert grid.n_pixels == 32
    assert grid.box_size == 32.0
    assert grid.pixel_size == 1.0
    assert grid.n_pixels_total == 1024
    
    # Test computed fields
    assert isinstance(grid.x_axis, torch.Tensor)
    assert isinstance(grid.y_axis, torch.Tensor)
    assert isinstance(grid.x_pixels, torch.Tensor)
    assert isinstance(grid.y_pixels, torch.Tensor)
    
    # Test shapes
    assert grid.x_axis.shape == (32,)
    assert grid.y_axis.shape == (32,)
    assert grid.x_pixels.shape == (32, 32)
    assert grid.y_pixels.shape == (32, 32)

def test_repr():
    grid = SquaredCartesianGrid2D(n_pixels=16, box_size=16.0, endpoint=False)
    repr_str = repr(grid)
    expected_str = "SquaredCartesianGrid2D(n_pixels=16, pixel_size=1.0, box_size=16.0, endpoint=False)"
    assert repr_str == expected_str

def test_invalid_inputs():
    # Test negative n_pixels
    with pytest.raises(ValueError, match="n_pixels must be positive"):
        SquaredCartesianGrid2D(n_pixels=-32, box_size=2.0)
        
    # Test zero n_pixels
    with pytest.raises(ValueError, match="n_pixels must be positive"):
        SquaredCartesianGrid2D(n_pixels=0, box_size=2.0)
        
    # Test negative pixel_size
    with pytest.raises(ValueError, match="box_size must be positive"):
        SquaredCartesianGrid2D(n_pixels=32, box_size=-1.0)
        
    # Test zero pixel_size
    with pytest.raises(ValueError, match="box_size must be positive"):
        SquaredCartesianGrid2D(n_pixels=32, box_size=0.0)

def test_to_device_and_dtype():
    grid = SquaredCartesianGrid2D(n_pixels=32, box_size=2.0)
    
    # Test dtype conversion
    grid.to(dtype=torch.float64, device=torch.device('cpu'))
    assert grid.x_axis.dtype == torch.float64
    assert grid.y_axis.dtype == torch.float64
    assert grid.x_pixels.dtype == torch.float64
    assert grid.y_pixels.dtype == torch.float64
    
    # Test device placement (only if CUDA available)
    if torch.cuda.is_available():
        grid.to(dtype=torch.float32, device=torch.device('cuda'))
        assert grid.x_axis.device.type == 'cuda'
        assert grid.y_axis.device.type == 'cuda'
        assert grid.x_pixels.device.type == 'cuda'
        assert grid.y_pixels.device.type == 'cuda'

@mark.parametrize("endpoint", [True, False])
def test_grid_values(endpoint: bool):
    grid = SquaredCartesianGrid2D(n_pixels=3, box_size=3.0, endpoint=endpoint)
    if endpoint:
        expected_range = [-1.5, 0.0, 1.5]  # For 3 pixels with endpoint
    else:
        expected_range = [-1.5, -0.5, 0.5]  # For 3 pixels without endpoint
    
    # Test axis values approximately
    np.testing.assert_array_almost_equal(grid.x_axis.numpy(), expected_range)
    np.testing.assert_array_almost_equal(grid.y_axis.numpy(), expected_range)