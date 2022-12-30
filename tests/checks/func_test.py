from typing import Any, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import Protocol

from eincheck.checks.func import check_func
from eincheck.types import _ShapeType
from tests.utils import Dummy, arr, raises_literal


def test_single_arg() -> None:
    @check_func("*x i i -> *x")
    def trace(x: Any) -> Any:
        return x.diagonal(axis1=-1, axis2=-2).sum(-1)

    trace(arr(3, 3))
    trace(arr(3, 7, 7))

    with raises_literal("x dim 1: expected i=4 got 5"):
        trace(arr(4, 5))

    goofy_array = Dummy(
        shape=(3, 3),
        diagonal=lambda **k: Dummy(shape=(3, 3), sum=lambda _: Dummy(shape=(3,))),
    )
    with raises_literal("output0: expected rank 0, got shape (3,)"):
        trace(goofy_array)


def test_two_args() -> None:
    @check_func("..., *x 2 -> *x")
    def foo(x: Any, y: Any) -> Any:
        return y[..., 0] + y[..., 1] + x

    foo(arr(5), arr(5, 2))
    foo(arr(), arr(5, 2))
    foo(2, arr(5, 2))

    with raises_literal("y: expected rank at least 1, got shape ()"):
        foo(arr(), arr())

    with raises_literal("output0: expected rank 1, got shape (10, 5)"):
        foo(arr(10, 1), arr(5, 2))


def test_no_args() -> None:
    @check_func("->i  i")
    def foo() -> Any:
        return arr(3, 3)

    foo()

    @check_func("-> i i")
    def bar(i: int, j: int) -> Any:
        return arr(i, j)

    bar(7, 7)

    with raises_literal("output0 dim 1: expected i=5 got 8"):
        bar(5, 8)


def test_var_arg() -> None:
    @check_func("*x -> *x _")
    def stack(*x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.stack(x, -1)

    for i in range(1, 4):
        stack(*(arr(4, 5) for _ in range(i)))

    with raises_literal("x_1: expected rank 2, got shape (10,)"):
        stack(arr(2, 3), arr(10), arr(2, 3))

    with raises_literal("x_2 dims (0, 1): expected x=(2, 3) got (2, 5)"):
        stack(arr(2, 3), arr(2, 3), arr(2, 5))


def test_var_kwarg() -> None:
    @check_func("i j -> i j")
    def foo(**kwargs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        x = sum(len(k) * v for k, v in kwargs.items())
        assert isinstance(x, np.ndarray)
        return x

    foo(x=arr(3, 4), y=arr(3, 4), z=arr(3, 4))

    with raises_literal("z dim 1: expected j=4 got 5"):
        foo(x=arr(3, 4), y=arr(3, 4), z=arr(3, 5))


def test_multiple_outputs() -> None:
    class _TensorWithSum(Protocol[_ShapeType]):
        shape: _ShapeType

        def sum(self, dims: Tuple[int, ...]) -> "_TensorWithSum[Any]":
            ...

    TensorWithSum = _TensorWithSum[Any]

    @check_func("i j k -> i, j, k")
    def partial_sums(
        argument: TensorWithSum,
    ) -> Tuple[TensorWithSum, TensorWithSum, TensorWithSum]:
        return (argument.sum((1, 2)), argument.sum((0, 2)), argument.sum((0, 1)))

    partial_sums(arr(3, 4, 5))

    with raises_literal("argument: expected rank 3, got shape (7, 42)"):
        partial_sums(arr(7, 42))

    class DummyArray:
        def __init__(self, shape: Tuple[int, int, int], bad_dim: int):
            self.shape = shape
            self.bad_dim = bad_dim

        def sum(self, dims: Tuple[int, ...]) -> npt.NDArray[Any]:
            (remaining_dim,) = {0, 1, 2} - set(dims)
            return arr(self.shape[remaining_dim] + (remaining_dim == self.bad_dim))

    for bad_dim, dim_string in enumerate("ijk"):
        with raises_literal(
            f"output{bad_dim} dim 0: expected {dim_string}={bad_dim + 10} "
            f"got {bad_dim + 11}"
        ):
            partial_sums(DummyArray((10, 11, 12), bad_dim))


def test_default_args() -> None:
    @check_func("i, j k, i, j *w -> k *w")
    def foo(
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64] = arr(7, 42),
        *,
        z: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64] = arr(7, 1),
    ) -> Any:
        a: npt.NDArray[np.float64] = x * z
        b = y.reshape(y.shape + (1,) * (len(w.shape) - 1)) * w[:, None]
        return a.sum() * b.sum(0)

    foo(arr(3), z=arr(3))
    foo(arr(3), arr(7, 10), z=arr(3))
    foo(arr(3), y=arr(7, 10), z=arr(3))
    foo(arr(3), z=arr(3), w=arr(7, 3, 5))
    foo(arr(3), arr(7, 10), z=arr(3), w=arr(7, 3, 5))
    foo(arr(3), y=arr(7, 10), z=arr(3), w=arr(7, 3, 5))
    foo(y=arr(7, 10), z=arr(3), x=arr(3), w=arr(7, 3, 5))

    with raises_literal("w dim 0: expected j=5 got 7"):
        foo(arr(4), arr(5, 9), z=arr(4))

    with raises_literal("z dim 0: expected i=5 got 7"):
        foo(arr(5), z=arr(7))

    with raises_literal("y: expected rank 2, got shape (4,)"):
        foo(x=arr(4), y=arr(4), z=arr(4))

    with raises_literal("w dim 0: expected j=7 got 2"):
        foo(arr(1), z=arr(1), w=arr(2))


@pytest.mark.parametrize("use_tup", [True, False])
def test_collections(use_tup: bool) -> None:
    @check_func("i, i -> i")
    def bar(x: Sequence[Any], y: Sequence[Any]) -> Sequence[Any]:
        if use_tup:
            return tuple(zip(x, y))
        else:
            return list(zip(x, y))

    bar([1, 2, 3], (0, 1, 2))
    bar([], [])
    bar((), [])

    with raises_literal("y dim 0: expected i=2 got 3"):
        bar([True, False], (3, 2, 1))


def test_invalid_usage() -> None:
    with raises_literal("Expected at least 3 input parameters, got 2"):

        @check_func("i, j, k -> i j k")
        def foo(a: Any, b: Any) -> Any:
            pass

    @check_func("i -> i, i")
    def bar(x: Any) -> Any:
        return x

    with raises_literal("Expected 2 outputs, got 1"):
        bar(arr(4))
