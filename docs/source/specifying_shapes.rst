Specifying Shapes
=================

There are several ways to specify a shape constraint (see the ``ShapeArg`` type alias).
The most common way is a single string with a list of dimension constraints separated by spaces.
The syntax for shape constraints is a superset of the einstein notation used for ops like ``einsum``.
The following sections describe this syntax.

Integers and Variables
----------------------

The simplest way to specify a dimension constraint is a literal integer.

Variables can be used to capture dynamic dimensions.
Variables can be any string of ``[a-zA-Z]+``.

For example, the shape spec ``"i 4 inner"`` will match any 3D Tensor where the second dimension is 4, and bind ``i`` to the first dimension and ``inner`` to the last.

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> check_shapes(
    ...     (np.random.randn(3, 4, 5), "i 4 inner"),
    ... )
    {'i': 3, 'inner': 5}
    >>> check_shapes(
    ...     (np.random.randn(3, 4), "i 4 inner"),
    ... )
    Traceback (most recent call last):
        ...
    ValueError: arg0: expected rank 3, got shape (3, 4)
      arg0: got (3, 4) expected [i 4 inner]

    >>> check_shapes(
    ...     (np.random.randn(3, 4, 5), "i 4 i"),
    ... )
    Traceback (most recent call last):
        ...
    ValueError: arg0 dim 2: expected i=3 got 5
        i=3
      arg0: got (3, 4, 5) expected [i 4 i]



Variables have a single meaning across all the specs being considered.
The shape spec ``"i i"`` matches any 2D Tensor with the same size of both dimensions (i.e. a square matrix).
Similarly, ``check_shapes((x, "i j"), (y, "i k"))`` will check that ``x`` and ``y`` are both 2D Tensors and that ``x.shape[0] == y.shape[0]``, and will return ``{"j": x.shape[1], "k": y.shape[1]}``.

Expressions
-----------

In addition to literal integers and variables, a dimension constraint can also be an expression of integers, variables, and binary operators.
Expressions are always enclosed in parentheses, and let us capture more complex shape relations.

There are three available operators for integers: addition (``+``), subtraction (``-``), and multiplication (``*``).

For example,

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> x = np.random.randn(3, 5)
    >>> y = np.random.randn(3, 7)
    >>>
    >>> check_shapes(
    ...     (x, "n x"),
    ...     (y, "n y"),
    ...     (np.concatenate([x, y], axis=1), "n (x + y)"),
    ...     ((x[:, :, None] + y[:, None, :]).reshape(3, -1), "n (x * y)"),
    ... )
    {'n': 3, 'x': 5, 'y': 7}
    >>> check_shapes((x, "n ((2 * n) - 1)"))
    {'n': 3}

Repeated Dimension Constraints
------------------------------

A dimension constraint can also match against multiple dimensions in the tensor's shape.
If the dimension constraint is followed by a ``*`` it is a repeated dimension constraint, and matches zero or more dimensions in the tensor shape.

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> check_shapes(
    ...     (np.random.randn(4), "3* x"),
    ...     (np.random.randn(3, 4), "3* x"),
    ...     (np.random.randn(3, 3, 3, 3, 4), "3* x"),
    ... )
    {'x': 4}
    >>> check_shapes(
    ...     (np.random.randn(2, 1, 1, 4), "2 i* (4 * i)"),
    ...     (np.random.randn(2, 2, 2), "(i + 1)*"),
    ... )
    {'i': 1}
    >>> check_shapes(
    ...     (np.random.randn(7, 7, 1, 7), "i*")
    ... )
    Traceback (most recent call last):
        ...
    ValueError: arg0 dim 2: expected i=7 got 1
        i=7
      arg0: got (7, 7, 1, 7) expected [i*]

Variadic Dimension Constraints
------------------------------

A dimension constraint with a ``*`` in front of it is variadic. Variadic dimension constraints evaluate to a tuple instead of a single integer and match multiple dimensions in the tensor's shape.

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> x = np.random.randn(3, 4, 5, 6)
    >>>
    >>> check_shapes((x, "*i"))
    {'i': (3, 4, 5, 6)}
    >>> check_shapes((x, "3 *i 6"))
    {'i': (4, 5)}
    >>> check_shapes(
    ...     (x, "3 *i x"),
    ...     (np.random.randn(4, 4), "*i"),
    ... )
    Traceback (most recent call last):
        ...
    ValueError: arg1 dims (0, 1): expected i=(4, 5) got (4, 4)
        i=(4, 5)
        x=6
      arg0: got (3, 4, 5, 6) expected [3 *i x]
      arg1: got (4, 4)       expected [*i]

Trying to mix tuple-valued variables and integer-valued variables will cause errors.

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> check_shapes((np.random.randn(3, 4, 5), "*2"))
    Traceback (most recent call last):
        ...
    ValueError: arg0: expected variadic DimSpec *2 to evaluate to a tuple, got 2
      arg0: got (3, 4, 5) expected [*2]
    >>> check_shapes(
    ...     (np.random.randn(3, 4, 5), "*i"),
    ...     (np.random.randn(6), "i"),
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Found variables in both variadic and non-variadic expressions: i

There are two binary operator on tuple-valued expressions: broadcast (``^``) and concat (``||``).
Broadcasting follows `numpy-style broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

.. csv-table:: Tuple operators examples
   :header: "``i``", "``j``", "``(i ^ j)``", "``(i || j)``"

    "(2, 1)", "(1, 3)", "(2, 3)", "(2, 1, 1, 3)"
    "(4,)", "(3, 4)", "(3, 4)", "(4, 3, 4)"
    "(4, 2)", "(7, 1, 2)", "(7, 4, 2)", "(4, 2, 7, 1, 2)"

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> x = np.random.randn(3, 1, 5)
    >>> y = np.random.randn(5, 5)
    >>>
    >>> check_shapes(
    ...     (x, "*x 5"),
    ...     (y, "*y 5"),
    ...     (x + y, "*(x ^ y) 5"),
    ... )
    {'x': (3, 1), 'y': (5,)}


Underscores and Ellipses
------------------------

An underscore (``_``) will match a single dimension of any size.
An ellispe (``...``) will match multiple dimensions of any size.
Repeated underscores (``_*``) is equivalent to an ellipse.

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> x = np.random.randn(3, 1, 5)
    >>>
    >>> check_shapes((x, "i _ 5"))
    {'i': 3}
    >>> check_shapes((x, "i _ _"))
    {'i': 3}
    >>> check_shapes((x, "... 5"))
    {}
    >>> check_shapes((x, "3 1 ... 5"))
    {}
    >>> check_shapes((x, "_* 5"))
    {}

Limitations
-----------

In order to compare a shape to a shape spec, eincheck needs to be able to determine which dimensions correspond to which dimension specs.
This means there can be at most one dimension constraint that matches an unknown number of dimensions: ellipses, repeated dimension constraints, and variadic dimension constraints with unassigned variables.

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> x = np.random.randn(3, 5, 2, 2)
    >>>
    >>> check_shapes((x, "*i *j"))
    Traceback (most recent call last):
        ...
    ValueError: Unable to determine bindings for: arg0
      arg0: got (3, 5, 2, 2) expected [*i *j]
    >>> check_shapes((x, "... 2*"))
    Traceback (most recent call last):
        ...
    ValueError: Unable to determine bindings for: arg0
      arg0: got (3, 5, 2, 2) expected [_* 2*]
    >>>
    >>> # These are ok because j is already assigned.
    >>> check_shapes((x, "*i *j"), j=(2, 2))
    {'j': (2, 2), 'i': (3, 5)}
    >>> check_shapes(
    ...     (x, "*i *j"),
    ...     (x[0, 0], "*j"),
    ... )
    {'j': (2, 2), 'i': (3, 5)}

Eincheck is not a general equation solver.
To determine the value of a variable, there must be a dimension spec that is just  that variable.
Eincheck will reorder the Tensors to determine variable values first.

.. doctest::

    >>> import numpy as np
    >>> from eincheck import check_shapes
    >>>
    >>> check_shapes(
    ...     (np.random.randn(4, 2), "(2 * i) i"),
    ... )
    {'i': 2}
    >>> check_shapes(
    ...     (np.random.randn(4, 2), "(i + 1) (i - 1)"),
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Unable to check: [arg0] missing variables: [i]
      arg0: got (4, 2) expected [(i+1) (i-1)]
    >>> check_shapes(
    ...     (np.random.randn(4, 2), "(i + 1) (i - 1)"),
    ...     i=3,
    ... )
    {'i': 3}
    >>> check_shapes(
    ...     (np.random.randn(4, 2), "(i + 1) (i - 1)"),
    ...     (np.random.randn(3), "i"),
    ... )
    {'i': 3}