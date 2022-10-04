

def to_flat_index(index, shape, order='C'):
    # Compute cumulative size
    ndim = len(shape)
    csize = [1]
    for x in shape[::-1]:
        csize.append(x*csize[-1])
    csize = csize[:-1]
    if order == 'C':
        csize.reverse()
    elif order != 'F':
        raise ValueError('Order can be C or F.')

    # Compute flat position
    pos = 0
    for i in range(ndim):
        pos += index[i]*csize[i]
    
    return pos
        

def to_shape_index(pos, shape, order='C'):
    # Compute cumulative size
    ndim = len(shape)
    csize = [1]
    for x in shape[::-1]:
        csize.append(x*csize[-1])
    csize = csize[:-1]
    if order == 'C':
        csize.reverse()
    elif order != 'F':
        raise ValueError('Order can be C or F.')
        
    # Compute new index
    index = list(shape)
    if order == 'C':
        for i in range(ndim):
            index[i] = int(pos/csize[i])
            pos = pos % csize[i]
    elif order == 'F':
        for i in range(ndim - 1, -1, -1):
            index[i] = int(pos/csize[i])
            pos = pos % csize[i]
        
    return index
