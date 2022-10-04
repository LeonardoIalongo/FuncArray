from funcarray import array
import numpy as np
from funcarray.utils import to_shape_index
from funcarray.utils import to_flat_index

N = 10
range_arr = np.arange(N**2).reshape((N, N))


class TestReshape():
    def test_indexing_coherence(self):
        shape = (3, 4, 5, 6)
        # C ordering
        for index in np.ndindex(shape):
            pos = to_flat_index(index, shape, order='C')
            new_index = tuple(to_shape_index(pos, shape, order='C'))
            assert index == new_index

        # F ordering
        for index in np.ndindex(shape):
            pos = to_flat_index(index, shape, order='F')
            new_index = tuple(to_shape_index(pos, shape, order='F'))
            assert index == new_index

    # def test_1d_to_2d(self):
    #     def foo(i):
    #         return float(i)

    #     a = array(N**2, foo)
    #     assert np.all(a.reshape((N, N)).to_numpy() == range_arr)


class TestCompletion():
    def test_zero_fill(self):
        def foo(i, j):
            return 0.0

        a = array((N, N), foo)
        assert np.all(a.to_numpy() == np.zeros((N, N)))

    def test_ones_fill(self):
        def foo(i, j):
            return 1.0

        a = array((N, N), foo)
        assert np.all(a.to_numpy() == np.ones((N, N)))

    def test_range_fill(self):
        def foo(i, j):
            return float(i*N + j)

        a = array((N, N), foo)
        assert np.all(a.to_numpy() == range_arr)
