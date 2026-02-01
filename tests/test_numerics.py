import unittest
import numpy as np
from cubed_sphere.numerics.spectral import lgl_diff_matrix, lgl_nodes_weights

class TestNumerics(unittest.TestCase):
    def test_lgl_diff_constant_function(self):
        """Test that derivative of a constant function is zero."""
        N = 8
        D = lgl_diff_matrix(N)
        
        # Constant calculation: f(x) = 1
        f = np.ones(N)
        df = D @ f
        
        # Assert close to zero
        np.testing.assert_allclose(df, np.zeros(N), atol=1e-14)

    def test_lgl_diff_linear_function(self):
        """Test that derivative of x is 1."""
        N = 8
        D = lgl_diff_matrix(N)
        x, _ = lgl_nodes_weights(-1, 1, N)
        
        # f(x) = x
        f = x
        df = D @ f
        
        # Assert close to 1
        expected = np.ones(N)
        np.testing.assert_allclose(df, expected, atol=1e-13)

if __name__ == "__main__":
    unittest.main()
