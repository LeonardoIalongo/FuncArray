from funcarray import array
import pytest
import numpy as np
from funcarray.utils import to_shape_index
from funcarray.utils import to_flat_index
import re

N = 10
range_arr = np.arange(N**2).reshape((N, N))


class TestReshape():
    def test_wrong_order(self):
        with pytest.raises(ValueError, match='Order can be C or F, not D.'):
            to_flat_index((0, 1), (3, 2), order='D')
        with pytest.raises(ValueError, match='Order can be C or F, not 0.'):
            to_shape_index(4, (3, 2), order=0)

    def test_zero_index(self):
        shape = (5, 6)
        index = (0, 0)

        # C ordering
        pos = to_flat_index(index, shape, order='C')
        assert pos == 0
        new_index = to_shape_index(pos, shape, order='C')
        assert index == new_index

        # F ordering
        pos = to_flat_index(index, shape, order='F')
        assert pos == 0
        new_index = to_shape_index(pos, shape, order='F')
        assert index == new_index

    def test_full_index(self):
        shape = (7, 2, 3, 1)
        index = (6, 1, 2, 0)

        # C ordering
        pos = to_flat_index(index, shape, order='C')
        assert pos == np.prod(shape) - 1
        new_index = to_shape_index(pos, shape, order='C')
        assert index == new_index

        # F ordering
        pos = to_flat_index(index, shape, order='F')
        assert pos == np.prod(shape) - 1
        new_index = to_shape_index(pos, shape, order='F')
        assert index == new_index

    def test_out_of_bounds_index(self):
        shape = (2, 3, 1)
        index = (0, 1, 2)

        msg = 'Index {} out of bounds for shape {}.'.format(index, shape)
        # C ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            pos = to_flat_index(index, shape, order='C')
        # F ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            pos = to_flat_index(index, shape, order='F')

        pos = 6
        msg = 'Position {} out of bounds for shape {}.'.format(pos, shape)
        # C ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            new_index = to_shape_index(pos, shape, order='C')
        # F ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            new_index = to_shape_index(pos, shape, order='F')

    def test_indexing_coherence_vector(self):
        shape = (5, )
        # C ordering
        for index in np.ndindex(shape):
            pos = to_flat_index(index, shape, order='C')
            new_index = to_shape_index(pos, shape, order='C')
            assert index == new_index

        # F ordering
        for index in np.ndindex(shape):
            pos = to_flat_index(index, shape, order='F')
            new_index = to_shape_index(pos, shape, order='F')
            assert index == new_index

    def test_indexing_coherence(self):
        shape = (3, 4, 5, 6)
        # C ordering
        for index in np.ndindex(shape):
            pos = to_flat_index(index, shape, order='C')
            new_index = to_shape_index(pos, shape, order='C')
            assert index == new_index

        # F ordering
        for index in np.ndindex(shape):
            pos = to_flat_index(index, shape, order='F')
            new_index = to_shape_index(pos, shape, order='F')
            assert index == new_index


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
