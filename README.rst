=================
Functional Arrays
=================

The funcarray package contains a class to construct arrays that generate 
values on demand, with optimised functions to iterate over it and perform 
specific operations.

* Free software: GNU General Public License v3


Installation
------------
Install using:

.. code-block:: python

   pip install funcarray

Usage
-----
Functional arrays are instantiated by passing it a shape and a function that 
must return each element of the matrix given the index of the element and a set
of arguments. The given function must therefore obey the format ``fun(*index, *args)``.

.. code-block:: python

    from funcarray import array
    from numba import jit
    import numpy as np
    from numpy.random import default_rng

    @jit(nopython=True)
    def fun(i, j, x, y):
        return x[i]*y[j]

    N = 10000
    rnd = default_rng()
    x = rnd.random(N, dtype='f8')
    y = rnd.random(N, dtype='f8')
    a = array((N, N), fun, x, y)

    # Compute sum over all elements
    print(a.sum())

Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature

To build a development environment run:

.. code-block:: bash

    python3 -m venv env 
    source env/bin/activate 
    pip install -e '.[dev]'

For testing:

.. code-block:: bash

    pytest --cov

Credits
-------
This is a project by `Leonardo Niccolò Ialongo <https://datasciencephd.eu/students/leonardo-niccol%C3%B2-ialongo/>`_ .

