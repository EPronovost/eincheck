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


Data Objects
------------

A dollar sign (``$``) can be used with data objects decorated with ``check_data``.
For example, the following two ``check_shapes`` are equivalent.

.. doctest::

    >>> import numpy as np
    >>> import numpy.typing as npt
    >>> from eincheck import check_shapes, check_data
    >>> from typing import NamedTuple
    >>> from numpy.random import randn
    >>>
    >>> @check_data(x="i", y="i")
    ... class Foo(NamedTuple):
    ...     x: npt.NDArray[float]
    ...     y: npt.NDArray[float]
    ...
    >>> f = Foo(randn(3), randn(3))
    >>> z = randn(3, 3)
    >>> check_shapes(
    ...     **{
    ...         "f.x": (f.x, "i"),
    ...         "f.y": (f.y, "i"),
    ...         "z": (z, "i i"),
    ...     }
    ... )
    {'i': 3}
    >>> check_shapes(f=(f, "$"), z=(z, "i i"))
    {'i': 3}

See the API section on this decorator for more info.

Sometimes it is easier to specify the shapes of individual fields inside a data object.
When using ``check_shapes``, users can explicitly access these fields (e.g. ``f.x`` in the example above).

When using ``check_func`` and ``check_data``, dot name paths can be used to access subfields of an object, regardless of whether the object is decorated with ``check_data``.
As dots are not valid in Python identifiers, dictionaries are currently needed to use such names.

..  doctest::

    >>> import numpy as np
    >>> import numpy.typing as npt
    >>> from eincheck import check_func
    >>> from typing import NamedTuple
    >>> from numpy.random import randn
    >>>
    >>> class Foo(NamedTuple):
    ...     x: npt.NDArray[float]
    ...     y: npt.NDArray[float]
    ...
    >>> @check_func(**{"a.x": "i", "a.y": "j", "b": "i j"})
    ... def func(a: Foo, b: npt.NDArray[float]) -> npt.NDArray[float]:
    ...     return a.x[:, None] * a.y + b
    ...
    >>> func(Foo(randn(3), randn(4)), randn(3, 4)).shape
    (3, 4)
    >>> func(Foo(randn(3), randn(4)), randn(2, 4))
    Traceback (most recent call last):
        ...
    ValueError: b dim 0: expected i=3 got 2
        i=3
        j=4
      a.x: got (3,)   expected [i]
      a.y: got (4,)   expected [j]
      b: got (2, 4) expected [i j]
    >>>
    >>> # Equivalent, using integer indices instead of named fields.
    >>> @check_func(**{"a.0": "i", "a.1": "j", "b": "i j"})
    ... def func2(a: Foo, b: npt.NDArray[float]) -> npt.NDArray[float]:
    ...     return a.x[:, None] * a.y + b
    ...
    >>> func2(Foo(randn(3), randn(4)), randn(3, 4)).shape
    (3, 4)
    >>> func2(Foo(randn(3), randn(4)), randn(2, 4))
    Traceback (most recent call last):
        ...
    ValueError: b dim 0: expected i=3 got 2
        i=3
        j=4
      a.0: got (3,)   expected [i]
      a.1: got (4,)   expected [j]
      b: got (2, 4) expected [i j]


Dot name paths can be particularly useful when working with subfields that are themselves decorated with ``check_data``.
Using ``$`` enforces that all shape variables match, which is sometimes not desired.

.. doctest::

    >>> import numpy
    >>> import numpy.typing as npt
    >>> from eincheck import check_data
    >>> from dataclasses import dataclass
    >>>
    >>> @check_data(tokens="n t d", mask="n t")
    ... @dataclass
    ... class TokensWithMask:
    ...     tokens: npt.NDArray[float]
    ...     mask: npt.NDArray[float]
    ...
    ...     @staticmethod
    ...     def rand(n: int, t: int, d: int) -> "TokensWithMask":
    ...         return TokensWithMask(np.random.randn(n, t, d), np.random.rand(n, t) > 0.3)
    ...
    >>> # With this decorator, the t dimension of query, key, and value has to match.
    >>> @check_data(query="$", key="$", value="$")
    ... @dataclass
    ... class AttentionData1:
    ...     query: TokensWithMask
    ...     key: TokensWithMask
    ...     value: TokensWithMask
    ...
    >>> q = TokensWithMask.rand(3, 4, 5)
    >>> k = TokensWithMask.rand(3, 7, 5)
    >>> _ = AttentionData1(q, q, q)
    >>> _ = AttentionData1(q, k, k)
    Traceback (most recent call last):
        ...
    ValueError: key.tokens dim 1: expected t=4 got 7
        n=3
        t=4
        d=5
      query.tokens: got (3, 4, 5) expected [n t d]
      query.mask: got (3, 4)    expected [n t]
      key.tokens: got (3, 7, 5) expected [n t d]
      key.mask: got (3, 7)    expected [n t]
      value.tokens: got (3, 7, 5) expected [n t d]
      value.mask: got (3, 7)    expected [n t]
    >>>
    >>> # Using dot name paths allows for different sequence dimensions.
    >>> @check_data({"query.tokens": "n q d", "key.tokens": "n k d", "value.tokens": "n k d"})
    ... @dataclass
    ... class AttentionData2:
    ...     query: TokensWithMask
    ...     key: TokensWithMask
    ...     value: TokensWithMask
    ...
    >>> _ = AttentionData2(q, q, q)
    >>> _ = AttentionData2(q, k, k)
    >>> _ = AttentionData2(q, k, TokensWithMask.rand(3, 7, 2))
    Traceback (most recent call last):
        ...
    ValueError: value.tokens dim 2: expected d=5 got 2
        n=3
        q=4
        d=5
        k=7
      query.tokens: got (3, 4, 5) expected [n q d]
      key.tokens: got (3, 7, 5) expected [n k d]
      value.tokens: got (3, 7, 2) expected [n k d]


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