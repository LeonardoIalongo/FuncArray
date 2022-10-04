

def to_flat_index(index, shape, order='C'):
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
