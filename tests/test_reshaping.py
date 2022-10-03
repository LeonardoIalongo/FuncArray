from funcarray import array
from numba import njit
import numpy as np

N = 10
range_arr = np.arange(N**2).reshape((N, N))


class TestCompletion():
    def test_zero_fill(self):
        @njit
        def foo(i, j):
            return 0.0

        a = array((N, N), foo)
        assert np.all(a.to_numpy() == np.zeros((N, N)))

    def test_ones_fill(self):
        @njit
        def foo(i, j):
            return 1.0

        a = array((N, N), foo)
        assert np.all(a.to_numpy() == np.ones((N, N)))

    def test_range_fill(self):
        @njit
        def foo(i, j):
            return float(i*N + j)

        a = array((N, N), foo)
        assert np.all(a.to_numpy() == range_arr)
