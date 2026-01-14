import unittest
import torch
from cryolike.util.math import absq, complex_mul_real

class TestMath(unittest.TestCase):

    def test_absq(self):
        """Test the absq function for real and complex tensors."""
        # Test with a real tensor
        real_tensor = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        expected_real = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(absq(real_tensor), expected_real))

        # Test with a complex tensor
        complex_tensor = torch.tensor([1.0 + 2.0j, -3.0 - 4.0j], dtype=torch.complex64)
        # Expected: (1^2 + 2^2), ((-3)^2 + (-4)^2) -> 5, 25
        expected_complex = torch.tensor([5.0, 25.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(absq(complex_tensor), expected_complex))

        # Test with a multi-dimensional tensor
        md_tensor = torch.tensor([[1.0, -2.0], [0.0, 5.0]], dtype=torch.float64)
        expected_md = torch.tensor([[1.0, 4.0], [0.0, 25.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(absq(md_tensor), expected_md))

        # Test with a zero tensor
        zero_tensor = torch.zeros(2, 2, dtype=torch.complex128)
        expected_zero = torch.zeros(2, 2, dtype=torch.float64)
        self.assertTrue(torch.allclose(absq(zero_tensor), expected_zero))

    def test_complex_mul_real(self):
        """Test the complex_mul_real function."""
        # Test with two complex tensors
        c1 = torch.tensor([1.0 + 2.0j, 3.0 - 1.0j], dtype=torch.complex64)
        c2 = torch.tensor([2.0 + 3.0j, -1.0 + 2.0j], dtype=torch.complex64)
        # Expected: (1*2 - 2*3), (3*(-1) - (-1)*2) -> -4, -1
        expected_cc = torch.tensor([-4.0, -1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(complex_mul_real(c1, c2), expected_cc))

        # Test with two real tensors
        r1 = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        r2 = torch.tensor([4.0, 5.0, -6.0], dtype=torch.float32)
        expected_rr = torch.tensor([4.0, -10.0, -18.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(complex_mul_real(r1, r2), expected_rr))

        # Test with one complex and one real tensor
        c_tensor = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64)
        r_tensor = torch.tensor([2.0, -3.0], dtype=torch.float32)
        # Expected: real(c_tensor) * r_tensor -> (1*2), (3*(-3)) -> 2, -9
        expected_cr = torch.tensor([2.0, -9.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(complex_mul_real(c_tensor, r_tensor), expected_cr))

        # Test with one real and one complex tensor
        # Expected: r_tensor * real(c_tensor) -> (2*1), (-3*3) -> 2, -9
        self.assertTrue(torch.allclose(complex_mul_real(r_tensor, c_tensor), expected_cr))

        # Test with different dtypes
        c_double = torch.tensor([1.0 + 2.0j], dtype=torch.complex128)
        r_double = torch.tensor([3.0], dtype=torch.float64)
        expected_double = torch.tensor([3.0], dtype=torch.float64)
        self.assertTrue(torch.allclose(complex_mul_real(c_double, r_double), expected_double))

if __name__ == '__main__':
    unittest.main()
