import numpy as np
from .utils import to_shape_index
from .utils import to_flat_index
from math import ceil


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
            yield self[r]

    def __getitem__(self, index):
        # Ensure index is compatible with shape
        if isinstance(index, int) or isinstance(index, slice):
            index = (index,)

        # If index is missing dimensions, assume all elements are selected
        if len(index) > self.ndim:
            raise IndexError('Index has {} dimensions instead of {}.'.format(
                len(index), self.ndim))

        for i in range(self.ndim - len(index)):
            index += (slice(None, None, None), )

        # Ensure all elements of index are int or slice
        if not np.all([isinstance(i, int) or isinstance(i, slice) 
                      for i in index]):
            raise IndexError('Index must be int or slice.')
        
        # If any index is a slice return new object
        if np.any([isinstance(i, slice) for i in index]):
            # Create new array function with changed indexing
            tmp_f = self.fun
            tmp_shape = self.shape

            # Define new shape
            new_shape = tuple()
            for n, i in enumerate(index):
                if isinstance(i, slice):
                    if i.start is None:
                        start = 0
                    else:
                        start = i.start
                        if start < 0:
                            start += self.shape[n]
                    if i.stop is None:
                        stop = self.shape[n]
                    else:
                        stop = i.stop
                        if stop < 0:
                            stop += self.shape[n]
                    if i.step is None:
                        step = 1
                    else:
                        step = i.step
                    elem = ceil(abs((stop - start)/step))
                    new_shape += (elem, )

            def new_fun(*nargs):
                ndim = len(new_shape)
                new_index = nargs[:ndim]
                args = nargs[ndim:]
                if isinstance(new_index, int):
                    new_index = (new_index, )
                
                # Convert new index to old one
                prev_index = tuple()
                n = 0
                for m, i in enumerate(index):
                    if isinstance(i, int):
                        prev_index += (i,)
                    else:
                        if i.step is None:
                            step = 1
                        else:
                            step = i.step
                        if i.start is None:
                            if step < 0:
                                start = -1
                            else:
                                start = 0
                        else:
                            start = i.start
                        
                        pos = start + new_index[n]*step
                        if pos < 0:
                            pos += tmp_shape[m]
                        prev_index += (pos, )
                        n += 1

                return tmp_f(*prev_index, *args)

            return self.__class__(
                new_shape, new_fun, *self.args, order=self.order)

        # else return the queried value
        else:
            if np.any([(i >= s) or (-i > s) 
                      for i, s in zip(index, self.shape)]):
                msg = 'Index {} out of bounds for shape {}.'.format(
                    index, self.shape)
                raise IndexError(msg)

            # If index is negative convert it to positive equivalent
            index = tuple([s + i if i < 0 else i 
                           for i, s in zip(index, self.shape)])

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
