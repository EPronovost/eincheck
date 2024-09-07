import enum
from typing import Any, Callable, Dict, NamedTuple, Protocol, Sequence, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from eincheck.checks.func import check_func, check_func2
from eincheck.types import _ShapeType
from tests.testing_utils import Dummy, arr, raises_literal

_T_Callable = TypeVar("_T_Callable", bound=Callable[..., Any])


@enum.unique
class DecoratorMode(enum.Enum):
    V1 = "v1"
    V2 = "v2"
    V2_ARROW = "v2 arrow"

    def str_decorator(self, spec: str) -> Callable[[_T_Callable], _T_Callable]:
        if self is DecoratorMode.V1:
            return check_func(spec)
        if self is DecoratorMode.V2_ARROW:
            return check_func2(spec)
        if self is DecoratorMode.V2:
            in_spec, out_spec = spec.split("->")
            return check_func2(in_spec, out_spec)
        raise ValueError(f"Unknown enum value: {self}")

    def kwarg_decorator(
        self, items: Dict[str, str], out_str: str = ""
    ) -> Callable[[_T_Callable], _T_Callable]:
        if self is DecoratorMode.V1:
            return check_func(out_str, **items)
        if self is DecoratorMode.V2 or self is DecoratorMode.V2_ARROW:
            return check_func2(items, out_str)
        raise ValueError(f"Unknown enum value: {self}")


@pytest.fixture(params=DecoratorMode)
def mode(request: Any) -> DecoratorMode:
    x = request.param
    assert isinstance(x, DecoratorMode)
    return x


def test_single_arg(mode: DecoratorMode) -> None:
    @mode.str_decorator("*x i i -> *x")
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


def test_two_args(mode: DecoratorMode) -> None:
    @mode.str_decorator("..., *x 2 -> *x,")
    def foo(x: Any, y: Any) -> Any:
        return y[..., 0] + y[..., 1] + x

    foo(arr(5), arr(5, 2))
    foo(arr(), arr(5, 2))
    foo(2, arr(5, 2))

    with raises_literal("y: expected rank at least 1, got shape ()"):
        foo(arr(), arr())

    with raises_literal("output0: expected rank 1, got shape (10, 5)"):
        foo(arr(10, 1), arr(5, 2))


def test_no_args(mode: DecoratorMode) -> None:
    @mode.str_decorator("->i  i")
    def foo() -> Any:
        return arr(3, 3)

    foo()

    @mode.str_decorator("-> i i")
    def bar(i: int, j: int) -> Any:
        return arr(i, j)

    bar(7, 7)

    with raises_literal("output0 dim 1: expected i=5 got 8"):
        bar(5, 8)


def test_var_arg(mode: DecoratorMode) -> None:
    @mode.str_decorator("*x -> *x _")
    def stack(*x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.stack(x, -1)

    for i in range(1, 4):
        stack(*(arr(4, 5) for _ in range(i)))

    with raises_literal("x_1: expected rank 2, got shape (10,)"):
        stack(arr(2, 3), arr(10), arr(2, 3))

    with raises_literal("x_2 dims (0, 1): expected x=(2, 3) got (2, 5)"):
        stack(arr(2, 3), arr(2, 3), arr(2, 5))

    @mode.kwarg_decorator({"x.0": "i", "x.1": "j i"})
    def foo(*x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return sum(x)  # type: ignore[return-value]

    foo(arr(4), arr(5, 4))
    foo(arr(4), arr(5, 4), arr(6, 5, 4))
    foo(arr(4), arr(5, 4), arr(6, 5, 4), arr(7, 6, 5, 4))

    with raises_literal("x.1 dim 1: expected i=3 got 2"):
        foo(arr(3), arr(2, 2))


def test_var_kwarg(mode: DecoratorMode) -> None:
    @mode.str_decorator("i j -> i j")
    def foo(**kwargs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        x = sum(len(k) * v for k, v in kwargs.items())
        assert isinstance(x, np.ndarray)
        return x

    foo(x=arr(3, 4), y=arr(3, 4), z=arr(3, 4))

    with raises_literal("z dim 1: expected j=4 got 5"):
        foo(x=arr(3, 4), y=arr(3, 4), z=arr(3, 5))

    @mode.kwarg_decorator({"kwargs.a": "i", "kwargs.b": "i"})
    def bar(**kwargs: Any) -> Any:
        return kwargs["a"] * kwargs["b"]

    bar(a=arr(4), b=arr(4))
    bar(a=arr(4), b=arr(4), c=arr(7))

    with raises_literal("kwargs.b dim 0: expected i=3 got 5"):
        bar(a=arr(3), b=arr(5))

    with raises_literal("'b'", KeyError):
        bar(a=arr(2))


def test_multiple_outputs(mode: DecoratorMode) -> None:
    class _TensorWithSum(Protocol[_ShapeType]):
        shape: _ShapeType

        def sum(self, dims: Tuple[int, ...]) -> "_TensorWithSum[Any]": ...

    TensorWithSum = _TensorWithSum[Any]

    @mode.str_decorator("i j k -> i, j, k")
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

    @mode.str_decorator("i, j -> i j, i j")
    def foo(x: Any, y: Any) -> Any:
        a = x[:, None] + y[None, :]
        b = x[:, None] * y[None, :]
        if x.shape == y.shape:
            return a, b, x * y

        return a, b

    foo(arr(4), arr(7))
    foo(arr(7), arr(7))


def test_default_args(mode: DecoratorMode) -> None:
    @mode.str_decorator("i, j k, i, j *w -> k *w")
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
def test_collections(mode: DecoratorMode, use_tup: bool) -> None:
    @mode.str_decorator("i, i -> i")
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


def test_invalid_usage(mode: DecoratorMode) -> None:
    with raises_literal("Expected at least 3 input parameters, got 2"):

        @mode.str_decorator("i, j, k -> i j k")
        def foo(a: Any, b: Any) -> Any:
            pass

    @mode.str_decorator("i -> i, i")
    def bar(x: Any) -> Any:
        return x

    with raises_literal("Expected at least 2 outputs, got 1"):
        bar(arr(4))


def test_invalid_usage_v1() -> None:
    with raises_literal("Spec for bad specified in both args and kwargs"):

        @check_func("i -> i", bad="i")
        def biz(bad: Any) -> Any:
            return bad


def test_kwarg_specs_v1() -> None:
    @check_func("a, b -> a b d", d="d")
    def foo(a: Any, b: Any, c: Any, d: Any) -> Any:
        return a[:, None, None] + b[:, None] + d

    foo(arr(4), arr(5), arr(6), arr(7))
    foo(arr(4), d=arr(5), c=arr(6), b=arr(7))

    with raises_literal("a: expected rank 1, got shape (2, 3)"):
        foo(arr(2, 3), arr(1), arr(1), arr(1))

    with raises_literal("d: expected rank 1, got shape (2, 3)"):
        foo(arr(1), arr(1), arr(1), arr(2, 3))


@pytest.mark.parametrize("use_arrow", [True, False])
def test_no_input_args_v1(use_arrow: bool) -> None:
    @check_func(("->" if use_arrow else "") + "i j, j k", x="i j")
    def foo(x: Any, y: Any) -> Any:
        return x + y, x.T @ y

    foo(arr(42, 7), arr(42, 7))

    with raises_literal("output0 dim 1: expected j=1 got 5"):
        foo(arr(5, 1), arr(5, 5))


def test_methods(mode: DecoratorMode) -> None:
    class Foo:
        @mode.kwarg_decorator({"x": "i 7"})
        def __init__(self, x: Any):
            self.x = x

        @mode.str_decorator("_, i j -> ...")
        def m(self, y: Any) -> Any:
            return self.x + y

        @classmethod
        @mode.kwarg_decorator({"y": "i"}, "i")
        def c(cls, y: Any) -> Any:
            return y + y

        @staticmethod
        @mode.str_decorator("i j -> i j")
        def s(y: Any) -> Any:
            return -y + arr(5)

        @property
        @mode.kwarg_decorator({}, "i 7")
        def p(self) -> Any:
            return 2 * self.x

    with raises_literal("x dim 1: expected 7=7 got 6"):
        Foo(arr(1, 6))

    with raises_literal("x: expected rank 2, got shape (1, 2, 3)"):
        Foo(arr(1, 2, 3))

    foo = Foo(arr(3, 7))

    foo.m(arr(1, 7))

    with raises_literal("y: expected rank 2, got shape (7, 8, 9)"):
        foo.m(arr(7, 8, 9))

    foo.c(arr(5))
    Foo.c(arr(42))

    with raises_literal("y: expected rank 1, got shape (7, 7)"):
        foo.c(arr(7, 7))

    with raises_literal("output0 dim 0: expected i=3 got 6"):
        Foo.c((True, True, False))

    foo.s(arr(3, 5))
    Foo.s(arr(5, 5))

    with raises_literal("output0 dim 1: expected j=1 got 5"):
        foo.s(arr(3, 1))

    _ = foo.p

    foo.x = arr(5, 5)
    with raises_literal("output0 dim 1: expected 7=7 got 5"):
        _ = foo.p


def test_extra_names(mode: DecoratorMode) -> None:
    with raises_literal(
        "Parameter names not found in function signature: {'c'}", NameError
    ):

        @mode.kwarg_decorator(dict(a="x", b="x", c="x"))
        def foo1(a: Any, b: Any) -> None:
            pass

    with raises_literal(
        "Parameter names not found in function signature: {'c'}", NameError
    ):

        @mode.kwarg_decorator({"a.y": "x", "b": "x", "c.x": "x"})
        def foo2(a: Any, b: Any) -> None:
            pass

    # This is ok with **kwargs.
    @mode.kwarg_decorator(dict(a="x", b="x", c="x"))
    def foo3(a: Any, b: Any, **kwargs: Any) -> None:
        pass

    @mode.kwarg_decorator({"a.x": "foo", "c.first": "foo"})
    def foo4(a: Any, b: Any, **kwargs: Any) -> None:
        pass


def test_name_path(mode: DecoratorMode) -> None:

    class Foo(NamedTuple):
        x: Any
        y: Any

    @mode.kwarg_decorator({"f.x": "i", "f.y": "i j", "z": "j"})
    def foo(f: Foo, z: Any) -> Any:
        return f.x[:, None] + f.y * z

    foo(Foo(arr(4), arr(4, 3)), arr(3))

    with raises_literal("f.y dim 0: expected i=3 got 4"):
        foo(Foo(arr(3), arr(4, 5)), arr(5))


def test_output_dict_v2() -> None:

    @check_func2("", {"0": "i", "1": "i i"})
    def output_tuple(x: int) -> Tuple[Any, Any]:
        return arr(7), arr(7, x)

    output_tuple(7)

    with raises_literal("output 1 dim 1: expected i=7 got 8"):
        output_tuple(8)

    @check_func2("", {"x": "i", "y": "i"})
    def output_dict(x: int) -> Dict[str, Any]:
        return {"x": arr(x), "y": arr(42)}

    output_dict(42)

    with raises_literal("output y dim 0: expected i=3 got 42"):
        output_dict(3)

    @check_func2("i, j", {"": "(2*i)"})
    def output_array(x: Any, y: Any) -> Any:
        return np.concatenate([x, y], 0)

    output_array(arr(5), arr(5))

    with raises_literal("output  dim 0: expected (2*i)=10 got 11"):
        output_array(arr(5), arr(6))

    @check_func2("i", {"x.0": "i", "x.1": "i"})
    def output_optional(x: Any) -> Dict[str, Any]:
        if x.shape[0] % 3 == 0:
            y = x
        elif x.shape[0] % 3 == 1:
            y = x[:-1]
        else:
            y = None

        return dict(x=(x, y))

    output_optional(arr(9))
    output_optional(arr(8))

    with raises_literal("output x.1 dim 0: expected i=7 got 6"):
        output_optional(arr(7))


def test_output_optional(mode: DecoratorMode) -> None:
    @mode.str_decorator("i -> i, i")
    def foo(x: Any) -> Tuple[Any, Any]:
        if x.shape[0] % 2 == 0:
            y = x
        else:
            y = None

        return x, y

    foo(arr(8))
    foo(arr(9))
