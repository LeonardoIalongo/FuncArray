from numba import jit
import numpy as np


class array(object):
    """The functional array is an array whose elements are computed on demand 
    rather than stored. 

    This allows for faster computation of properties of measures computed by 
    iterating over its elements. It also allows to have arrays that would not 
    fit in memory but can be handled as if they were.
    """
    def __init__(self, shape, fun, *args, dtype='f8'):
        """Return a FuncArray object. 

        :param fun: function that computes each element of the array.
        :type T: function.
        :param args: arguments necessary for fun to be computed.
        """

        self.fun = fun
        self.args = args
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def __iter__(self):
        for r in range(self.shape[0]):
            yield self[r]

    def __getitem__(self, index):
        if not isinstance(index, int):
            for s in index:
                if isinstance(s, slice):
                    raise ValueError('Slicing not yet supported.')
            return self.fun(*index, *self.args)

    def sum(self):
        return self._sum(self.fun, self.shape, self.args)

    @staticmethod
    @jit(nopython=True)
    def _sum(fun, shape, args):
        res = 0
        for index in np.ndindex(shape):
            res += fun(*index, *args)
        return res

    def to_numpy(self):
        return self._to_numpy(
            self.fun, self.shape, self.args, dtype=self.dtype)

    @staticmethod
    @jit(nopython=True)
    def _to_numpy(fun, shape, args, dtype):
        res = np.empty(shape, dtype=dtype)
        for index in np.ndindex(shape):
            res[index] = fun(*index, *args)
        return res
