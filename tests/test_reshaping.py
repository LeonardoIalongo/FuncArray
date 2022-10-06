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

        # C order
        assert np.all(a.reshape((N, N)).to_numpy() == range_arr)

        # F order
        assert np.all(a.reshape((N, N), order='F').to_numpy() == range_arr.T)

        # Test in place reshape
        a.shape = (N, N)
        assert np.all(a.to_numpy() == range_arr)

    def test_2d_to_1d(self):
        # C order
        def foo(i, j):
            return float(i*N + j)
        a = array((N, N), foo)
        assert np.all(a.reshape(N**2).to_numpy() == np.arange(N**2))

        # Test in place reshape
        a.shape = N**2
        assert np.all(a.to_numpy() == np.arange(N**2))

        # F order
        def fii(i, j):
            return float(i + j*N)
        a = array((N, N), fii)
        assert np.all(a.reshape(N**2, order='F').to_numpy() == np.arange(N**2))

    def test_1d_coherence(self):
        def foo(i):
            return float(i)
        a = array(N**2, foo)

        # C order
        assert np.all(a.reshape((N, N)).reshape(N**2).to_numpy() 
                      == np.arange(N**2))

        # F order
        assert np.all(a.reshape((N, N), order='F')
                       .reshape(N**2, order='F').to_numpy() 
                      == np.arange(N**2))

    def test_2d_coherence(self):
        def foo(i, j):
            return float(i*N + j)
        a = array((N, N), foo)

        # C order
        assert np.all(a.reshape((N/2, N*2)).reshape((N, N)).to_numpy() 
                      == a.to_numpy())

        # F order
        assert np.all(a.reshape((N/2, N*2), order='F')
                       .reshape((N, N), order='F').to_numpy() 
                      == a.to_numpy())

    def test_2d_to_3d(self):
        def foo(i, j):
            return float(i*N + j)
        a = array((N, N), foo)

        # C order
        assert np.all(a.reshape((N/2, N/2, 4)).reshape((N, N)).to_numpy() 
                      == a.to_numpy())

        # F order
        assert np.all(a.reshape((N/2, N/2, 4), order='F')
                       .reshape((N, N), order='F').to_numpy() 
                      == a.to_numpy())


class TestSlicing():
    def test_single_element_1d(self):
        def foo(i):
            return float(i)
        a = array(N**2, foo)

        assert np.all([a[i] == i for i in range(N**2)])
        assert np.all([a[i-N**2] == i for i in range(N**2)])

    def test_single_element_2d(self):
        def foo(i, j):
            return float(i*N + j)
        a = array((N, N), foo)

        assert np.all([a[i, j] == i*N + j for i in range(N) for j in range(N)])
        assert np.all([a[i-N, j-N] == i*N + j 
                      for i in range(N) for j in range(N)])

    def test_single_element_3d(self):
        def foo(i, j, k):
            return float(i*N**2 + j*N + k)
        a = array((N, N, N), foo)

        assert np.all([a[i, j, k] == i*N**2 + j*N + k 
                      for i in range(N) for j in range(N) for k in range(N)])
        assert np.all([a[i-N, j-N, k-N] == i*N**2 + j*N + k 
                      for i in range(N) for j in range(N) for k in range(N)])

    def test_range_1d(self):
        def foo(i):
            return float(i)
        a = array(N**2, foo)

        assert np.all([a[2:5].to_numpy() == np.arange(2, 5)])
        assert np.all([a[:5].to_numpy() == np.arange(5)])
        assert np.all([a[2:].to_numpy() == np.arange(2, N**2)])
        assert np.all([a[:].to_numpy() == np.arange(N**2)])
        assert np.all([a[2:50:4].to_numpy() == np.arange(2, 50, 4)])

    def test_range_2d(self):
        def foo(i, j):
            return float(i*N + j)
        a = array((N, N), foo)

        assert np.all(a[0, 2:5].to_numpy() == range_arr[0, 2:5])
        assert np.all(a[0:3, 5].to_numpy() == range_arr[0:3, 5])
        assert np.all(a[0:3, 5:6].to_numpy() == range_arr[0:3, 5:6])
        assert np.all(a[:3, 5:].to_numpy() == range_arr[:3, 5:])
        assert np.all(a[:, :].to_numpy() == range_arr)
        assert np.all(a[0:6:2, 3:7:3].to_numpy() == range_arr[0:6:2, 3:7:3])
        assert np.all(a[0].to_numpy() == range_arr[0])
        assert np.all(a[:, 0].to_numpy() == range_arr[:, 0])

    def test_range_3d(self):
        def foo(i, j, k):
            return float(i*N**2 + j*N + k)
        a = array((N, N, N), foo)

        test_arr = np.arange(N**3).reshape((N, N, N))

        assert np.all(a[0, 0, 2:5].to_numpy() == test_arr[0, 0, 2:5])
        assert np.all(a[0, 2:5, 0].to_numpy() == test_arr[0, 2:5, 0])
        assert np.all(a[2:5, 0, 0].to_numpy() == test_arr[2:5, 0, 0])
        assert np.all(a[0, 2:5, :3].to_numpy() == test_arr[0, 2:5, :3])
        assert np.all(a[0, 2:, :3].to_numpy() == test_arr[0, 2:, :3])
        assert np.all(a[:, 2:, :3].to_numpy() == test_arr[:, 2:, :3])
        assert np.all(a[3].to_numpy() == test_arr[3])
        assert np.all(a[3, 2].to_numpy() == test_arr[3, 2])
        assert np.all(a[:, 3, 2].to_numpy() == test_arr[:, 3, 2])
        assert np.all(a[::3, 2::2, :8:4].to_numpy() 
                      == test_arr[::3, 2::2, :8:4])

    def test_negative_range_1d(self):
        def foo(i):
            return float(i)
        a = array(N**2, foo)

        assert np.all([a[-5:-2].to_numpy() == np.arange(N**2)[-5:-2]])
        assert np.all([a[-2:-5:-1].to_numpy() == np.arange(N**2)[-2:-5:-1]])
        assert np.all([a[:-5].to_numpy() == np.arange(N**2)[:-5]])
        assert np.all([a[-2:].to_numpy() == np.arange(N**2)[-2:]])
        assert np.all([a[::-1].to_numpy() == np.arange(N**2)[::-1]])

    def test_negative_range_2d(self):
        def foo(i, j):
            return float(i*N + j)
        a = array((N, N), foo)

        assert np.all(a[0, -5:-2].to_numpy() == range_arr[0, -5:-2])
        assert np.all(a[0:3, -2:-5:-1].to_numpy() == range_arr[0:3, -2:-5:-1])
        assert np.all(a[:-3, -5:-8:-2].to_numpy() == range_arr[:-3, -5:-8:-2])
        assert np.all(a[::-1, ::-1].to_numpy() == range_arr[::-1, ::-1])

    def test_negative_range_3d(self):
        def foo(i, j, k):
            return float(i*N**2 + j*N + k)
        a = array((N, N, N), foo)

        test_arr = np.arange(N**3).reshape((N, N, N))

        assert np.all(a[0, 0, -5:-2].to_numpy() == test_arr[0, 0, -5:-2])
        assert np.all(a[0, -2:-5:-1, 0].to_numpy() == test_arr[0, -2:-5:-1, 0])
        assert np.all(a[2:5, :-3, -5:-8:-2].to_numpy() 
                      == test_arr[2:5, :-3, -5:-8:-2])
        assert np.all(a[-7:-5, :-3, -5:-8:-2].to_numpy() 
                      == test_arr[-7:-5, :-3, -5:-8:-2])
        assert np.all(a[::-1, ::-1, ::-1].to_numpy() 
                      == test_arr[::-1, ::-1, ::-1])

    def test_out_of_bounds_slice_1d(self):
        def foo(i):
            return float(i)
        a = array(N**2, foo)

        msg = 'Index (100, ) out of bounds for shape (100, ).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a[100]
        msg = 'Index (-101, ) out of bounds for shape (100, ).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a[-101]

    def test_out_of_bounds_index_2d(self):
        def foo(i, j):
            return float(i*N + j)

        a = array((N, N), foo)

        msg = 'Index (10, 0) out of bounds for shape (10, 10).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a[10, 0]
        msg = 'Index (0, -11) out of bounds for shape (10, 10).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a[0, -11]

    def test_out_of_bounds_index_3d(self):
        def foo(i, j, k):
            return float(i*N**2 + j*N + k)

        a = array((N, N, N), foo)

        msg = 'Index (10, 0, 0) out of bounds for shape (10, 10, 10).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a[10, 0, 0]
        msg = 'Index (10, 100, 0) out of bounds for shape (10, 10, 10).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a[10, 100, 0]
        msg = 'Index (0, -11, 0) out of bounds for shape (10, 10, 10).'
        with pytest.raises(ValueError, match=re.escape(msg)):
            a[0, -11, 0]


class TestIteration():
    def test_1d_iter(self):
        def foo(i):
            return float(i)

        # C ordering
        a = array(N, foo, order='C')
        for x, y in zip(a, np.arange(N)):
            assert x == y
        
        # F ordering
        a = array(N, foo, order='F')
        for x, y in zip(a, np.arange(N)):
            assert x == y

    def test_2d_iter(self):
        def foo(i, j):
            return float(i*N + j)

        # C ordering
        a = array((N, N), foo, order='C')
        for x, y in zip(a, np.arange(N**2).reshape((N, N), order='C')):
            assert np.all(x == y)
        
        # F ordering
        a = array((N, N), foo, order='F')
        for x, y in zip(a, np.arange(N**2).reshape((N, N), order='F')):
            assert np.all(x == y)

    def test_3d_iter(self):
        def foo(i, j, k):
            return float(i*N**2 + j*N + k)

        # C ordering
        a = array((N, N, N), foo, order='C')
        for x, y in zip(a, np.arange(N**3).reshape((N, N, N), order='C')):
            assert np.all(x == y)
        
        # F ordering
        a = array((N, N, N), foo, order='F')
        for x, y in zip(a, np.arange(N**3).reshape((N, N, N), order='F')):
            assert np.all(x == y)

    def test_nested_iter(self):
        def foo(i, j, k):
            return float(i*N**2 + j*N + k)

        # C ordering
        a = array((N, N, N), foo, order='C')
        for x0, y0 in zip(a, np.arange(N**3).reshape((N, N, N), order='C')):
            for x1, y1 in zip(x0, y0):
                for x2, y2 in zip(x1, y1):
                    assert x2 == y2
        
        # F ordering
        a = array((N, N, N), foo, order='F')
        for x0, y0 in zip(a, np.arange(N**3).reshape((N, N, N), order='F')):
            for x1, y1 in zip(x0, y0):
                for x2, y2 in zip(x1, y1):
                    assert x2 == y2


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
