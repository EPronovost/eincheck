.. eincheck documentation master file, created by
   sphinx-quickstart on Mon Feb 13 22:20:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to eincheck's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   specifying_shapes
   api
   performance

Getting Started
---------------

To install eincheck, run

.. code-block:: shell

    pip install eincheck

eincheck is compatible with `numpy <https://numpy.org/>`_, `pytorch <https://pytorch.org/>`_, `tensorflow <https://www.tensorflow.org/>`_, `jax <https://jax.readthedocs.io/en/latest/index.html>`_, or any tensor object that has a ``shape`` field which returns a ``Sequence[int | None]``.
While none of these libraries are required to use eincheck, you will most likely want to install at least one of them.

How to use eincheck
-------------------

There are three key functions in `eincheck`, described in :ref:`API`:

* ``check_shapes``: compares tensors against shape specifications
* ``check_func``: decorates functions to add shape checks on the inputs and outputs
* ``check_data``: decorates classes to add shape checks to class fields

`eincheck` is inspired by Einstein notation, so basic functionality should be intuitive to anyone familiar with `einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_ or `einops <https://einops.rocks/>`_.

.. testcode::

    from typing import Any, NamedTuple

    import numpy as np
    import numpy.typing as npt
    from eincheck import check_func, check_data

    @check_func("*x -> *x")
    def softmax(x):
        y = np.exp(x - np.max(x))
        return y / y.sum()

    @check_data(tokens="n q d", scores="n q k")
    class AttentionOutputs(NamedTuple):
        tokens: npt.NDArray[Any]
        scores: npt.NDArray[Any]

    @check_func("n q c, n k c, n k d -> $")
    def attention(query, key, value):
        coeffs = np.einsum("n q c, n k c -> n q k", query, key)
        weights = softmax(coeffs / np.sqrt(query.shape[-1]))
        outputs = (
            np.expand_dims(weights, -1) *
            np.expand_dims(value, 1)
        ).sum(2)
        return AttentionOutputs(outputs, weights)

.. testcode::
    :hide:

    out, weights = attention(
        np.random.randn(7, 4, 10),
        np.random.randn(7, 5, 10),
        np.random.randn(7, 5, 8),
    )
    print(out.shape)
    print(weights.shape)

.. testoutput::
    :hide:

    (7, 4, 8)
    (7, 4, 5)

Resources
---------

* :ref:`Specifying Shapes` contains information on how to format shape specifications (e.g. ``"... i j"``)
* :ref:`API` contains information on the ``check_*`` functions
* :ref:`Performance` contains information on making code with shape checks run faster


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
