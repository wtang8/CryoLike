from unittest.mock import patch
from pytest import mark, raises
import torch

from cryolike.util import PrecisionLevel, get_epsilon, check_nufft_installed

def testget_epsilon():
    assert get_epsilon(PrecisionLevel.SINGLE, 1e-5) == 1e-5
    assert get_epsilon(PrecisionLevel.SINGLE, 1e-7) == 1e-6
    assert get_epsilon(PrecisionLevel.DOUBLE, 1e-10) == 1e-10
    assert get_epsilon(PrecisionLevel.DOUBLE, 1e-13) == 1e-12
    assert get_epsilon(PrecisionLevel.SINGLE, None) == 1e-6
    assert get_epsilon(PrecisionLevel.DOUBLE, None) == 1e-12


@patch('cryolike.util.nufft_checks.find_spec')
def testcheck_nufft_installed_cpu(mock_find_spec):
    # Test finufft is found on CPU
    mock_find_spec.return_value = True
    check_nufft_installed(torch.device('cpu'))
    mock_find_spec.assert_called_with('finufft')

    # Test finufft is not found on CPU
    mock_find_spec.return_value = None
    with raises(Exception, match="CPU is requested, but finufft is not installed."):
        check_nufft_installed(torch.device('cpu'))


@patch('cryolike.util.nufft_checks.find_spec')
def testcheck_nufft_installed_cuda(mock_find_spec):
    # Test cufinufft is found on CUDA
    mock_find_spec.return_value = True
    check_nufft_installed(torch.device('cuda'))
    mock_find_spec.assert_called_with('cufinufft')

    # Test cufinufft is not found on CUDA
    mock_find_spec.return_value = None
    with raises(Exception, match="CUDA is requested, but cufinufft is not installed."):
        check_nufft_installed(torch.device('cuda'))
