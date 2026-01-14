import torch
from pytest import mark
from unittest.mock import patch

from cryolike.util import get_device

DEVICES = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

class TestGetDevice:

    @mark.parametrize("device", DEVICES)
    def test_str_input(self, device):
        _device = get_device(device)
        assert isinstance(_device, torch.device)
        assert _device.type == device

    @mark.parametrize("device", DEVICES)
    def test_torch_device_input(self, device):
        _device = get_device(torch.device(device))
        assert isinstance(_device, torch.device)
        assert _device.type == device

    @patch('torch.cuda.is_available')
    def test_none_input(self, mock_is_available):
        mock_is_available.return_value = True
        assert get_device() == torch.device('cuda')
        mock_is_available.return_value = False
        assert get_device() == torch.device('cpu')

    @patch('torch.cuda.is_available')
    def test_cuda_input(self, mock_is_available):
        mock_is_available.return_value = True
        assert get_device(torch.device('cuda')) == torch.device('cuda')
        mock_is_available.return_value = False
        assert get_device(torch.device('cuda')) == torch.device('cpu')