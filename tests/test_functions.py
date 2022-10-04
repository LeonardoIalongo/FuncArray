from funcarray import array
import numpy as np


N = 10


class TestElementSum():
    def test_zero_sum(self):
        def foo(i, j):
            return 0.0

        a = array((N, N), foo)
        assert a.sum() == 0.0

    def test_ones_sum(self):
        def foo(i, j):
            return 1.0

        a = array((N, N), foo)
        assert a.sum() == N**2

    def test_range_sum(self):
        def foo(i, j):
            return float(i + j)

        a = array((N, N), foo)
        assert a.sum() == 2*N*np.sum(np.arange(N))
