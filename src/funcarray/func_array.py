from numba import jit
import numpy as np
from .utils import to_shape_index
from .utils import to_flat_index


class array(object):
    """The functional array is an array whose elements are computed on demand 
    rather than stored. 

    This allows for faster computation of properties of measures computed by 
    iterating over its elements. It also allows to have arrays that would not 
    fit in memory but can be handled as if they were.
    """
    def __init__(self, shape, fun, *args):
        """Return a FuncArray object. 

        :param fun: function that computes each element of the array.
        :type T: function.
        :param args: arguments necessary for fun to be computed.
        """

        self.fun = fun
        self.args = args
        if isinstance(shape, int):
            self._shape = (shape,)
        else:
            self._shape = tuple(shape)
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape)

    def __iter__(self):
        for r in range(self.shape[0]):
            yield self[r, :]

    def __getitem__(self, index):
        if not isinstance(index, int):
            for s in index:
                if isinstance(s, slice):
                    raise ValueError('Slicing not yet supported.')
            return self.fun(*index, *self.args)

    def set_shape(self, shape):
        """See `reshape`."""
        self.reshape(shape, copy=False).asformat(self.format)

    def get_shape(self):
        """Get shape of a array."""
        return self._shape

    shape = property(fget=get_shape, fset=set_shape)

    def reshape(self, shape, order='C'):
        """ Change shape of array without changing its data.
        
        :param shape: new shape of array which must be compatible with previous
        shape.
        :type shape: tuple of ints.
        :param order: C or Fortran-like index order.
        :type order: {'C', 'F'}, optional.
        """
        
        # Ensure shape is compatible
        if np.prod(shape) != self.size:
            raise ValueError('cannot reshape array of size {} into shape {}'
                             .format(self.size, shape))

        if shape == self.shape:
            return self

        # Create new funcarray with changed indexing
        @jit(nopython=True)
        def new_fun(index, args):
            prev_index = to_shape_index(
                to_flat_index(index, shape, order=order),
                self.shape, order)
            return self.fun(*prev_index, *args)

        return self.__class__(shape, new_fun, self.args)

    def sum(self):
        """ Sum all elements of the array.
        """
        return self._sum(self.fun, self.shape, self.args)

    @staticmethod
    @jit(nopython=True)
    def _sum(fun, shape, args):
        res = 0
        for index in np.ndindex(shape):
            res += fun(*index, *args)
        return res

    def to_numpy(self):
        """ Return a numpy copy of the array.
        """
        return self._to_numpy(
            self.fun, self.shape, self.args)

    @staticmethod
    @jit(nopython=True)
    def _to_numpy(fun, shape, args):
        res = np.empty(shape)
        for index in np.ndindex(shape):
            res[index] = fun(*index, *args)
        return res
