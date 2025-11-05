import pytest
import torch
import numpy as np
from cryolike.util.typechecks import ensure_positive

def test_ensure_positive_float():
    ensure_positive(1.0, "test float")
    with pytest.raises(ValueError):
        ensure_positive(0.0, "test float")
    with pytest.raises(ValueError):
        ensure_positive(-1.0, "test float")

def test_ensure_positive_int():
    ensure_positive(1, "test int") 
    with pytest.raises(ValueError):
        ensure_positive(0, "test int")
    with pytest.raises(ValueError):
        ensure_positive(-1, "test int")

def test_ensure_positive_numpy():
    ensure_positive(np.array([1.0, 2.0]), "test numpy")
    with pytest.raises(ValueError):
        ensure_positive(np.array([1.0, 0.0]), "test numpy")
    with pytest.raises(ValueError):
        ensure_positive(np.array([1.0, -1.0]), "test numpy")

def test_ensure_positive_torch():
    ensure_positive(torch.tensor([1.0, 2.0]), "test torch")
    with pytest.raises(ValueError):
        ensure_positive(torch.tensor([1.0, 0.0]), "test torch")
    with pytest.raises(ValueError):
        ensure_positive(torch.tensor([1.0, -1.0]), "test torch")