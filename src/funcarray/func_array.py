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
    def __init__(self, shape, fun, *args, order='C'):
        """Return a FuncArray object. 

        :param shape: number of elements of each dimension of the array.
        :type shape: tuple of ints.
        :param fun: function that computes each element of the array.
        :type fun: function.
        :param args: arguments necessary for fun to be computed.
        :type args: any, optional.
        :param order: C or Fortran-like index order, relevant only for reshape.
        :type order: {'C', 'F'}, optional.
        """

        self.fun = fun
        self.args = args
        self.order = order
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
        if isinstance(index, int):
            if self.ndim == 1:
                return self.fun(index, *self.args)

        for s in index:
            if isinstance(s, slice):
                raise ValueError('Slicing not yet supported.')
        return self.fun(*index, *self.args)

    def set_shape(self, shape):
        """See `reshape`."""
        new_array = self.reshape(shape)
        self.__dict__ = new_array.__dict__

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
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        
        # Ensure shape is compatible
        if np.prod(shape) != self.size:
            raise ValueError('Cannot reshape array of size {} into shape {}.'
                             .format(self.size, shape))

        if shape == self.shape:
            return self

        # Create new array function with changed indexing
        tmp_f = self.fun
        tmp_shape = self.shape

        def new_fun(*nargs):
            ndim = len(shape)
            index = nargs[:ndim]
            args = nargs[ndim:]
            prev_index = to_shape_index(
                to_flat_index(index, shape, order=order),
                tmp_shape, order=order)
            return tmp_f(*prev_index, *args)

        return self.__class__(shape, new_fun, *self.args, order=order)

    def to_numpy(self):
        """ Return a numpy copy of the array.
        """
        res = np.empty(self.shape)
        for index in np.ndindex(self.shape):
            res[index] = self.fun(*index, *self.args)
        return res

    def sum(self):
        """ Sum all elements of the array.
        """
        res = 0
        for index in np.ndindex(self.shape):
            res += self.fun(*index, *self.args)
        return res
