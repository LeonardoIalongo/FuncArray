import numpy as np


def to_flat_index(index, shape, order='C'):
    # Ensure index is compatible with shape
    if isinstance(index, int):
        index = (index,)
    
    if np.any([(i >= s) or (-i > s) for i, s in zip(index, shape)]):
        msg = 'Index {} out of bounds for shape {}.'.format(index, shape)
        raise IndexError(msg)

    # If index is negative convert it to positive equivalent
    index = tuple([s + i if i < 0 else i for i, s in zip(index, shape)])

    # Compute cumulative size
    ndim = len(shape)
    csize = [1]
    if order == 'C':
        csize = [1]
        for x in shape[::-1]:
            csize.append(x*csize[-1])
        csize = csize[:-1]
        csize.reverse()
        
    elif order == 'F':
        for x in shape:
            csize.append(x*csize[-1])
        csize = csize[:-1]
    else:
        raise ValueError('Order can be C or F, not {}.'.format(order))
    
    # Compute flat position
    pos = 0
    for i in range(ndim):
        pos += index[i]*csize[i]

    return pos
        

def to_shape_index(pos, shape, order='C'):
    # Ensure position is compatible with shape
    size = np.prod(shape)
    if (pos >= size) or (-pos > size):
        msg = 'Position {} out of bounds for shape {}.'.format(pos, shape)
        raise IndexError(msg)

    # If index is negative convert it to positive equivalent
    if pos < 0:
        pos = pos + size

    # Compute cumulative size
    ndim = len(shape)
    csize = [1]
    if order == 'C':
        csize = [1]
        for x in shape[::-1]:
            csize.append(x*csize[-1])
        csize = csize[:-1]
        csize.reverse()
        
    elif order == 'F':
        for x in shape:
            csize.append(x*csize[-1])
        csize = csize[:-1]
    else:
        raise ValueError('Order can be C or F, not {}.'.format(order))
        
    # Compute new index
    index = tuple()
    if order == 'C':
        for i in range(ndim):
            index += (int(pos/csize[i]),)
            pos = pos % csize[i]
    elif order == 'F':
        for i in range(ndim - 1, -1, -1):
            index = (int(pos/csize[i]),) + index
            pos = pos % csize[i]
        
    return index
