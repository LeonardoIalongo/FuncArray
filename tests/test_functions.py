from funcarray import array
from numba import njit


class TestElementSum():
    def test_zero_sum(self):
        @njit
        def foo(i, j):
            return 0.0

        a = array((10, 10), foo)
        assert a.sum() == 0.0
