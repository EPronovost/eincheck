API
===

.. testsetup::

    import numpy as np
    import numpy.typing as npt
    from numpy.random import randn
    from eincheck import (
      check_func, check_data, check_shapes,
      disable_checks, enable_checks,
      parser_cache_clear, parser_cache_info, parser_resize_cache,
    )
    from typing import NamedTuple
    import attrs


.. autofunction:: eincheck.check_shapes

.. autofunction:: eincheck.check_func

The ``check_func`` decorator adds shape checks to the inputs and outputs of the decorated function.
The simplest way to specify shapes is as a comma separated list of input shape specs, an arrow ``->``, and a comma separated list of output shape specs.

.. doctest::

    >>> @check_func("i, i -> i")
    ... def foo(x, y):
    ...     return x + y
    ...
    >>> foo(randn(4), randn(4)).shape
    (4,)
    >>> foo(randn(4), randn(3))
    Traceback (most recent call last):
    ...
    ValueError: y dim 0: expected i=4 got 3
        i=4
      x: got (4,) expected [i]
      y: got (3,) expected [i]

Input specs match function parameters in the order they're declared.
There need to be at least as many function parameters as input shape specs.

.. doctest::

    >>> @check_func("i -> i")
    ... def foo(x, y):
    ...     return x + y
    >>> foo(randn(4), randn(4)).shape
    (4,)
    >>> foo(randn(3, 4), randn(1))
    Traceback (most recent call last):
    ...
    ValueError: x: expected rank 1, got shape (3, 4)
      x: got (3, 4) expected [i]
    >>> foo(randn(3), randn(3, 3))
    Traceback (most recent call last):
    ...
    ValueError: output0: expected rank 1, got shape (3, 3)
      i = 3
      output0: got (3, 3) expected [i]
    >>> @check_func("i, i, i -> i")
    ... def foo(x):
    ...     return x
    Traceback (most recent call last):
    ...
    ValueError: Expected at least 3 input parameters, got 1


The shape spec for a variadic positional argument (e.g. ``*args``) or variadic keyword argument (e.g. ``**kwargs``) is compared against each Tensor matching that argument.

.. doctest::

    >>> @check_func("*x -> _ *x")
    ... def stack(*x):
    ...     return np.stack(x, 0)
    ...
    >>> stack(randn(3, 4), randn(3, 4)).shape
    (2, 3, 4)

    >>> stack(randn(3, 4), randn(3, 5), randn(3, 4))
    Traceback (most recent call last):
    ...
    ValueError: x_1 dims (0, 1): expected x=(3, 4) got (3, 5)
        x=(3, 4)
      x_0: got (3, 4) expected [*x]
      x_1: got (3, 5) expected [*x]
      x_2: got (3, 4) expected [*x]


The shapes of function inputs can also be specified with keyword arguments using the parameter name.

.. doctest::

    >>> @check_func("i -> i", y="i 2", z="i 3")
    ... def foo(x, y, *, z):
    ...     return x + y[:, 0] + z[:, 2]
    ...
    >>> foo(randn(3), randn(3, 2), z=randn(3, 3)).shape
    (3,)

    >>> foo(randn(3), randn(3, 3), z=randn(3, 3)).shape
    Traceback (most recent call last):
    ...
    ValueError: y dim 1: expected 2=2 got 3
        i=3
      x: got (3,)   expected [i]
      y: got (3, 3) expected [i 2]
      z: got (3, 3) expected [i 3]


If you want to only use keywords for inputs you can omit the ``"->"`` in the function spec string.
For example, ``@check_func("x")`` is equivalent to ``@check_func("-> x")``.

If you specify an input shape with a keyword it should not also be included in the positional shape specs.

.. doctest::

    >>> @check_func("i", x="i")
    ... def foo(x):
    ...     return x

    >>> @check_func("i -> i", x="i")
    ... def bad(x):
    ...     return x
    Traceback (most recent call last):
    ...
    ValueError: Spec for x specified in both args and kwargs.


Multiple output specs can be used if the function returns a tuple.

.. doctest::

    >>> @check_func("i j -> i, j")
    ... def split_sum(x):
    ...     return x.sum(1), x.sum(0)
    ...
    >>> [i.shape for i in split_sum(randn(3, 4))]
    [(3,), (4,)]


Similar to the positional input arguments, there can be more outputs than are captured in the spec.

.. doctest::

    >>> @check_func("i j -> i")
    ... def split_sum(x):
    ...     return x.sum(1), x.sum(0)
    ...
    >>> @check_func("i j -> i, j")
    ... def bad(x):
    ...     return x
    ...
    >>> bad(randn(2, 3))
    Traceback (most recent call last):
    ...
    ValueError: Expected at least 2 outputs, got 1


``check_func`` can also be used to decorate class methods.
The first argument to the function is ``self``.


.. doctest::

    >>> class Foo:
    ...     def __init__(self, x):
    ...         self.x = x
    ...
    ...     @check_func("_, i -> i")
    ...     def foo(self, y):
    ...         return self.x + y
    ...
    >>> f = Foo(randn(4))
    >>> f.foo(randn(4)).shape
    (4,)

    >>> f.foo(randn(1))
    Traceback (most recent call last):
    ...
    ValueError: output0 dim 0: expected i=1 got 4
        i=1
      output0: got (4,) expected [i]

.. autofunction:: eincheck.check_data

The ``@check_data`` decorator can add shape assertions to `NamedTuple <https://docs.python.org/3/library/typing.html?highlight=namedtuple#typing.NamedTuple>`_,  `dataclass <https://docs.python.org/3/library/dataclasses.html>`_, and `attrs <https://www.attrs.org/en/stable/index.html#>`_ classes.
Keyword arguments are matched against fields of the class.

.. doctest::

    >>> @check_data(x="*i", y="*i")
    ... class Foo(NamedTuple):
    ...     x: npt.NDArray[float]
    ...     y: npt.NDArray[float]
    ...
    >>> _ = Foo(randn(3, 4), randn(3, 4))
    >>> _ = Foo(randn(4, 5), randn(4))
    Traceback (most recent call last):
    ...
    ValueError: y: expected rank 2, got shape (4,)
      i = (4, 5)
      x: got (4, 5) expected [*i]
      y: got (4,)   expected [*i]


Not all fields of the object need shape specs.

.. doctest::

    >>> @check_data(x="i", y="i")
    ... class Foo(NamedTuple):
    ...     x: npt.NDArray[float]
    ...     y: npt.NDArray[float]
    ...     z: npt.NDArray[float]
    ...
    >>> _ = Foo(randn(4), randn(4), randn(4))
    >>> _ = Foo(randn(5), randn(5), randn(42))

A dictionary can also be used to specify shapes instead of keyword arguments.

.. doctest::

    >>> @check_data({"x": "*i", "y": "*i"})
    ... class Foo(NamedTuple):
    ...     x: npt.NDArray[float]
    ...     y: npt.NDArray[float]
    ...
    >>> _ = Foo(randn(3, 4), randn(3, 4))
    >>> _ = Foo(randn(4, 5), randn(4))
    Traceback (most recent call last):
    ...
    ValueError: y: expected rank 2, got shape (4,)
      i = (4, 5)
      x: got (4, 5) expected [*i]
      y: got (4,)   expected [*i]


What if you want to compare the shapes in a ``@check_data`` decorated object with other tensors?
The shape spec ``$`` will match ``@check_data`` decorated objects and include all the shapes from the object.
For example, the following two ``check_shapes`` are equivalent.

.. doctest::

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

This can also be used to pass ``@check_data`` decorated objects to functions and include them in other classes.

.. doctest::

    >>> @check_data(x="i", y="i")
    ... class Foo(NamedTuple):
    ...     x: npt.NDArray[float]
    ...     y: npt.NDArray[float]
    ...
    ...     @check_func("$, i -> i")
    ...     def method(self, z: npt.NDArray[float]) -> npt.NDArray[float]:
    ...         return self.y + z
    ...
    >>> f = Foo(randn(4), randn(4))
    >>> f.method(f.x).shape
    (4,)

    >>> f.method(randn(7))
    Traceback (most recent call last):
    ...
    ValueError: z dim 0: expected i=4 got 7
        i=4
      self.x: got (4,) expected [i]
      self.y: got (4,) expected [i]
      z: got (7,) expected [i]

    >>> @check_func("$, i -> $")
    ... def add_x(f: Foo, x: npt.NDArray[float]) -> Foo:
    ...     return Foo(f.x + x, f.y)
    ...
    >>> _ = add_x(f, randn(4))
    >>> add_x(f, randn(5))
    Traceback (most recent call last):
    ...
    ValueError: x dim 0: expected i=4 got 5
        i=4
      f.x: got (4,) expected [i]
      f.y: got (4,) expected [i]
      x: got (5,) expected [i]

    >>> @check_data(f="$", g="i i")
    ... @attrs.frozen
    ... class Bar:
    ...     f: Foo
    ...     g: npt.NDArray[float]
    ...
    >>> _ = Bar(f, randn(4, 4))
    >>> Bar(f, randn(5, 5))
    Traceback (most recent call last):
    ...
    ValueError: g dim 0: expected i=4 got 5
        i=4
      f.x: got (4,)   expected [i]
      f.y: got (4,)   expected [i]
      g: got (5, 5) expected [i i]


The ``@check_data`` decorator only checks the shapes on construction.
If you modify class members after creation it will not be checked.
You can use ``check_shapes`` to re-check a data object.

.. doctest::

    >>> @check_data(p="i 2")
    ... @attrs.define
    ... class Foo:
    ...     p: npt.NDArray[float]
    ...
    >>> f = Foo(randn(3, 2))
    >>> f.p = randn(4) # unchecked
    >>> check_shapes((f, "$"))
    Traceback (most recent call last):
    ...
    ValueError: arg0.p: expected rank 2, got shape (4,)
      arg0.p: got (4,) expected [i 2]


.. autofunction:: eincheck.disable_checks

Context manager to disable eincheck.
This can be used to make code run faster once you're confident the shapes are correct.
`check_shapes` will return an empty dictionary.

.. doctest::

    >>> with disable_checks():
    ...     # Eincheck is a no-op inside this context.
    ...     print(check_shapes((randn(2, 3), "i")))
    ...
    {}

.. autofunction:: eincheck.enable_checks

Context manager to enable eincheck (e.g. if inside a `disable_checks` context).

.. doctest::

    >>> with disable_checks():
    ...     with enable_checks():
    ...         check_shapes((randn(2, 3), "i"))
    ...
    Traceback (most recent call last):
    ...
    ValueError: arg0: expected rank 1, got shape (2, 3)
      arg0: got (2, 3) expected [i]

.. autofunction:: eincheck.parser_cache_clear

Clear the ``lru_cache`` for parsing shape strings.

.. autofunction:: eincheck.parser_cache_info

Get the ``lru_cache`` cache info for the parser cache.

.. doctest::

  >>> parser_cache_clear()
  >>> check_shapes((randn(2, 3), "a b"), (randn(3, 4), "b c"), (randn(2, 3), "a b"))
  {'a': 2, 'b': 3, 'c': 4}
  >>> parser_cache_info()
  CacheInfo(hits=1, misses=2, maxsize=128, currsize=2)

.. autofunction:: eincheck.parser_resize_cache

Reset the parser cache to a ``lru_cache`` with the given size.
This will clear the cache and change the ``maxsize`` field in ``CacheInfo``.

