import enum
import inspect
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import attrs
import pytest
from typing_extensions import Protocol

from eincheck.checks.data import check_data
from eincheck.checks.func import check_func
from eincheck.checks.shapes import check_shapes
from eincheck.parser.grammar import ShapeArg, create_shape_spec
from tests.utils import Dummy, arr, raises_literal


@enum.unique
class DataType(enum.Enum):
    NAMED_TUPLE = "NamedTuple"
    DATACLASS = "Dataclass"
    DATACLASS_POST_INIT = "DataclassPostInit"
    ATTRS = "Attrs"
    ATTRS_POST_INIT = "AttrsPostInit"


class FooFunc(Protocol):
    def __call__(self, x: Any, y: Any, z: Any) -> Any:
        ...


def get_datatype(
    dt: DataType, method_sig: Optional[str] = None, **kwargs: ShapeArg
) -> FooFunc:
    if dt is DataType.NAMED_TUPLE:

        @check_data(**kwargs)
        class Foo1(NamedTuple):
            x: Any
            y: Any
            z: Any

            if method_sig:

                @check_func(method_sig)
                def bar(self, a: Any) -> Any:
                    return a

        return Foo1

    elif dt is DataType.DATACLASS:

        @check_data(**kwargs)
        @dataclass
        class Foo2:
            x: Any
            y: Any
            z: Any

            if method_sig:

                @check_func(method_sig)
                def bar(self, a: Any) -> Any:
                    return a

        return Foo2

    elif dt is DataType.DATACLASS_POST_INIT:

        @check_data(**kwargs)
        @dataclass
        class Foo3:
            x: Any
            y: Any
            z: Any

            def __post_init__(self) -> None:
                assert any((self.x is not None, self.y is not None, self.z is not None))

            if method_sig:

                @check_func(method_sig)
                def bar(self, a: Any) -> Any:
                    return a

        return Foo3

    elif dt is DataType.ATTRS:

        @check_data(**kwargs)
        @attrs.define
        class Foo4:
            x: Any
            y: Any
            z: Any
            if method_sig:

                @check_func(method_sig)
                def bar(self, a: Any) -> Any:
                    return a

        return Foo4

    elif dt is DataType.ATTRS_POST_INIT:

        @check_data(**kwargs)
        @attrs.define
        class Foo5:
            x: Any
            y: Any
            z: Any

            def __attrs_post_init__(self) -> None:
                assert any((self.x is not None, self.y is not None, self.z is not None))

            if method_sig:

                @check_func(method_sig)
                def bar(self, a: Any) -> Any:
                    return a

        return Foo5


@pytest.fixture(params=DataType)
def data_type(request: Any) -> DataType:
    x = request.param
    assert isinstance(x, DataType)
    return x


def test_basic(data_type: DataType) -> None:
    spec = dict(x="i j", y="j k", z="j 2")
    FooXYZ = get_datatype(data_type, **spec)

    obj = FooXYZ(arr(3, 4), arr(4, 3), arr(4, 2))
    FooXYZ(arr(3, 4), arr(4, 3), None)

    with raises_literal("y dim 0: expected j=5 got 7"):
        FooXYZ(arr(3, 5), arr(7, 7), arr(5, 2))

    with raises_literal("z dim 1: expected 2=2 got 3"):
        FooXYZ(arr(1, 2), arr(2, 1), arr(2, 3))

    s = obj._get_shapes()
    assert set(s) == set(spec)
    for k in spec:
        assert s[k][1] == create_shape_spec(spec[k])


def test_variadic(data_type: DataType) -> None:
    FooProduct = get_datatype(data_type, z="*x  *y", x="*x", y="*y")

    FooProduct(arr(3), arr(7), arr(3, 7))
    foo = FooProduct(Dummy((3, 5)), None, arr(3, 5, 7))

    check_shapes(data=(foo, "$"))

    foo.x.shape += (1,)
    with raises_literal("data.z dims (0, 1, 2): expected x=(3, 5, 1) got (3, 5, 7)"):
        check_shapes(data=(foo, "$"))

    with raises_literal("z: expected rank 2, got shape (3, 5, 7)"):
        FooProduct(arr(3, 5), arr(), arr(3, 5, 7))

    with raises_literal("Unable to determine bindings for {'z'}"):
        FooProduct(None, None, arr(3, 5))


def test_partial(data_type: DataType) -> None:
    FooPartial = get_datatype(data_type, x="*batch i", y="*batch (i + 1)")

    FooPartial(arr(7, 3, 5), arr(7, 3, 6), arr(1, 2))
    FooPartial(y=arr(7, 3, 5), x=arr(7, 3, 4), z=arr(1, 2))
    FooPartial(x=arr(8, 8), y=None, z="hello, world")
    FooPartial([True, False], [0, 1, 2], "hello")

    with raises_literal("Unable to determine bindings for {'y'}"):
        FooPartial(None, arr(4), None)

    with raises_literal("Unable to determine bindings for {'y'}"):
        FooPartial("eincheck", arr(4), None)


def test_incorrect_usage(data_type: DataType) -> None:
    with raises_literal("No field found: [w]"):
        get_datatype(data_type, w=[3, 7])

    with raises_literal("No field found: [a w]"):
        get_datatype(data_type, w=[3, 7], a="i", x="j", y="k")

    @check_func("$ -> i")
    def foo(x: Any) -> Any:
        pass

    with raises_literal("x: spec $ specified, but no _get_shapes method was found."):
        foo((1, 2, 3))


def test_bad_data_type() -> None:
    with raises_literal("Unexpected data type", TypeError):

        @check_data(x="i")
        class Foo:
            def __init__(self, x: Any) -> None:
                self.x = x


def test_func_arg(data_type: DataType) -> None:
    Foo = get_datatype(data_type, x="*n i", y="*n j", z="(i+j)")

    foo = Foo(arr(3, 4, 5), arr(3, 4, 7), arr(12))

    @check_func("$ -> *n")
    def f1(foo: Any, a: Any) -> Any:
        return foo.x[..., 0] + foo.y[..., 0] + a

    f1(foo, arr(3, 4))

    with raises_literal("output0: expected rank 2, got shape (2, 3, 4)"):
        f1(foo, arr(2, 3, 4))

    @check_func("i, $ -> j")
    def f2(x: Any, data: Any) -> Any:
        a = data.x * x
        b = a[..., None] + data.y[..., None, :]
        return b.sum((0, 1))

    foo2 = Foo(arr(3, 5), arr(3, 7), arr(12))

    f2(arr(5), foo2)

    with raises_literal("data.x dim 1: expected i=4 got 5"):
        f2(arr(4), foo2)

    with raises_literal("output0: expected rank 1, got shape (5, 7)"):
        f2(arr(5), foo)


def test_func_output(data_type: DataType) -> None:
    Foo = get_datatype(data_type, x="*n i", y="*n j", z="(i+j)")

    @check_func("i -> $")
    def f1(i: Any, j: Any) -> Any:
        return Foo(x=arr(3, 4, 5), y=j, z=None)

    f1(arr(5), arr(3, 4, 1))

    with raises_literal("output0.x dim 2: expected i=1 got 5"):
        f1(arr(1), arr(3, 4, 2))

    with raises_literal("y: expected rank 3, got shape (2,)"):
        f1(arr(5), arr(2))


def test_nested(data_type: DataType) -> None:
    Foo = get_datatype(data_type, x="i j", y="j k", z="*z")
    Bar = get_datatype(data_type, x="$", y="i j k", z="*z")

    foo = Foo(arr(3, 5), arr(5, 7), arr(1, 2))
    Bar(foo, arr(3, 5, 7), arr(1, 2))

    with raises_literal("y dim 0: expected i=3 got 2"):
        Bar(foo, arr(2, 5, 7), arr())

    with raises_literal("z: expected rank 2, got shape (42,)"):
        Bar(foo, arr(3, 5, 7), arr(42))


def test_signature(data_type: DataType) -> None:
    Foo = get_datatype(data_type, x="i j", y="j k", z="*z")

    sig = inspect.signature(Foo)
    assert list(sig.parameters) == ["x", "y", "z"]


def test_method(data_type: DataType) -> None:
    Foo = get_datatype(data_type, method_sig="$, *z -> *z", z="*z")

    foo = Foo(None, None, arr(3, 4, 5))
    foo.bar(arr(3, 4, 5))

    with raises_literal("a dims (0, 1, 2): expected z=(3, 4, 5) got (2, 2, 2)"):
        foo.bar(arr(2, 2, 2))
