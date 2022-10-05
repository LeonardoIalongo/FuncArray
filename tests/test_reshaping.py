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
            to_flat_index(index, shape, order='C')
        # F ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_flat_index(index, shape, order='F')

        pos = 6
        msg = 'Position {} out of bounds for shape {}.'.format(pos, shape)
        # C ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_shape_index(pos, shape, order='C')
        # F ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_shape_index(pos, shape, order='F')

    def test_negative_out_of_bounds_index(self):
        shape = (2, 3, 1)
        index = (0, 1, -2)

        msg = 'Index {} out of bounds for shape {}.'.format(index, shape)
        # C ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_flat_index(index, shape, order='C')
        # F ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_flat_index(index, shape, order='F')

        index = (0, -4, 0)

        msg = 'Index {} out of bounds for shape {}.'.format(index, shape)
        # C ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_flat_index(index, shape, order='C')
        # F ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_flat_index(index, shape, order='F')

        pos = -7
        msg = 'Position {} out of bounds for shape {}.'.format(pos, shape)
        # C ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_shape_index(pos, shape, order='C')
        # F ordering    
        with pytest.raises(IndexError, match=re.escape(msg)):
            to_shape_index(pos, shape, order='F')

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

    def test_negative_index_vector(self):
        shape = (10, )
        # C ordering
        for index in range(shape[0]):
            neg_pos = index - shape[0]
            pos = to_flat_index((neg_pos, ), shape, order='C')
            assert pos == index
            new_index = to_shape_index(neg_pos, shape, order='C')
            assert (index, ) == new_index

        # F ordering
        for index in range(shape[0]):
            neg_pos = index - shape[0]
            pos = to_flat_index((neg_pos, ), shape, order='F')
            assert pos == index
            new_index = to_shape_index(neg_pos, shape, order='F')
            assert (index, ) == new_index

    def test_negative_index(self):
        shape = (7, 2, 3)
        size = np.prod(shape)

        # C ordering
        for index in np.ndindex(shape):
            neg_index = tuple([i - s for i, s in zip(index, shape)])
            pos = to_flat_index(neg_index, shape, order='C')
            new_index = to_shape_index(pos - size, shape, order='C')
            assert index == new_index

        # F ordering
        for index in np.ndindex(shape):
            neg_index = tuple([i - s for i, s in zip(index, shape)])
            pos = to_flat_index(neg_index, shape, order='F')
            new_index = to_shape_index(pos - size, shape, order='F')
            assert index == new_index

    def test_incompatible_shapes(self):
        def foo(i):
            return float(i)

        a = array(N**3, foo)
        msg = 'Cannot reshape array of size 1000 into shape (10, 10).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a.reshape((N, N))
        with pytest.raises(ValueError, match=re.escape(msg)):
            a.shape = (N, N)

    def test_1d_to_2d(self):
        def foo(i):
            return float(i)
        a = array(N**2, foo)
        assert np.all(a.reshape((N, N)).to_numpy() == range_arr)

        # Test in place reshape
        a.shape = (N, N)
        assert np.all(a.to_numpy() == range_arr)

    def test_2d_to_1d(self):
        def foo(i, j):
            return float(i*N + j)
        a = array((N, N), foo)
        assert np.all(a.reshape(N**2).to_numpy() == np.arange(N**2))

        # Test in place reshape
        a.shape = N**2
        assert np.all(a.to_numpy() == np.arange(N**2))


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
