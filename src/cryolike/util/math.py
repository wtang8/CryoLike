import torch


def absq(a: torch.Tensor):
    if torch.is_complex(a):
        return a.real ** 2 + a.imag ** 2
    return a.real ** 2


def complex_mul_real(a1: torch.Tensor, a2: torch.Tensor):
    if torch.is_complex(a1) and torch.is_complex(a2):
        return (a1.real * a2.real - a1.imag * a2.imag)
    return a1.real * a2.real