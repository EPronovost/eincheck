Performance
===========


.. testsetup::

    import numpy as np
    import numpy.typing as npt
    from numpy.random import randn
    from eincheck import (
      check_func, check_data, check_shapes,
      disable_checks, enable_checks,
      parser_cache_clear, parser_cache_info, parser_resize_cache,
    )
    from typing import List

Adding eincheck to a project introduces extra computations, which will have some latency impact.
This impact can be significantly mitigated by following best practices.

There are two major parts to performing shape checks with eincheck: parsing the user input into internal data structures and checking the tensor shapes against those data structures.
In most cases, parsing will take significantly more time than actually doing the checks.
Writing performant code with eincheck thus requires us to minimize the amount of parsing necessary.

Doing Less Parsing
------------------

There are several ways to reduce the amount of parsing.

Use Decorators
^^^^^^^^^^^^^^

The decorators ``check_data`` and ``check_func`` will parse the inputs once and then reuse them each time the data object/function is called.
These decorators should be used whenever possible.

Cached Parsing
^^^^^^^^^^^^^^
There are cases where the abovementioned decorators cannot be used.
``check_shapes`` uses an ``lru_cache`` to cache parsing, initialized with a default size of 128.
To achieve good cache utilization, prefer to use constant shape specs.
For example:

.. doctest::

    >>> parser_cache_clear()
    >>> parser_cache_info()
    CacheInfo(hits=0, misses=0, maxsize=128, currsize=0)
    >>> def bad(x: npt.NDArray[np.float64], inds: List[int]) -> npt.NDArray[np.float64]:
    ...     y = x[..., inds]
    ...     # Bad! The shape spec for y will change for different length inds.
    ...     check_shapes((x, "*i _"), (y, f"*i {len(inds)}"))
    ...     return y
    >>> _ = bad(randn(5, 10), [1])
    >>> _ = bad(randn(5, 10), [4, 2])
    >>> _ = bad(randn(5, 10), [0, 1, 2])
    >>> _ = bad(randn(5, 10), [7, 7, 7, 7, 7])
    >>> _ = bad(randn(5, 10), [3, 1, 4, 1, 5])
    >>> parser_cache_info()
    CacheInfo(hits=5, misses=5, maxsize=128, currsize=5)
    >>>
    >>> parser_cache_clear()
    >>> def good(x: npt.NDArray[np.float64], inds: List[int]) -> npt.NDArray[np.float64]:
    ...     y = x[..., inds]
    ...     # Good! The shape specs are constant.
    ...     check_shapes((x, "*i _"), (y, "*i n"), n=len(inds))
    ...     return y
    >>> _ = good(randn(5, 10), [1])
    >>> _ = good(randn(5, 10), [4, 2])
    >>> _ = good(randn(5, 10), [0, 1, 2])
    >>> _ = good(randn(5, 10), [7, 7, 7, 7, 7])
    >>> _ = good(randn(5, 10), [3, 1, 4, 1, 5])
    >>> parser_cache_info()
    CacheInfo(hits=8, misses=2, maxsize=128, currsize=2)

The functions ``parser_cache_info``, ``parser_cache_clear``, and ``parser_resize_cache`` can be used to monitor and adjust the caching behavior.

Skip Parsing
^^^^^^^^^^^^

If caching is not a viable option (e.g. due to memory constraints), another option to improve performance is to pass lists instead of strings as the shape specs to ``check_shapes``.
The ``ShapeArg`` type used as input to ``check_shapes`` includes ``Sequence[Union[DimSpec, str, int, None]]``.
If a sequence is provided, eincheck will skip the parser and build the internal data structures directly from this list.
As such, advanced parsing features are not supported with this method.
The strings in the sequence must be single variable names, with no parentheses or binary operators.

.. doctest::

    >>> def foo(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ...     check_shapes((x, [None, "i", "i"]))
    ...     return np.diagonal(x, axis1=1, axis2=2)
    >>> foo(randn(3, 4, 4)).shape
    (3, 4)
    >>> def bad(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ...     # Bad! Can't use binary operator + with list ShapeArg.
    ...     check_shapes((x, ["i", "i+1"]))
    ...     return x
    >>> bad(randn(3, 4)).shape
    Traceback (most recent call last):
    ...
    ValueError: Variable name should be a valid python name, got i+1

Disabling Checks
----------------

The most powerful tool to make code using eincheck run faster is to disable eincheck altogether.
For example, eincheck can be used while initially developing code and then disabled in optimized production environments.
The ``disable_checks`` and ``enable_checks`` context managers can be used to disable and re-enable eincheck within certain scopes.

