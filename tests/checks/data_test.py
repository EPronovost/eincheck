import enum
from dataclasses import dataclass
from typing import Any, NamedTuple

import attrs
import pytest
from typing_extensions import Protocol

from eincheck.checks.data import check_data
from eincheck.parser.grammar import ShapeArg
from tests.utils import arr, raises_literal


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


def get_datatype(dt: DataType, **kwargs: ShapeArg) -> FooFunc:
    if dt is DataType.NAMED_TUPLE:

        @check_data(**kwargs)
        class Foo1(NamedTuple):
            x: Any
            y: Any
            z: Any

        return Foo1

    elif dt is DataType.DATACLASS:

        @check_data(**kwargs)
        @dataclass
        class Foo2:
            x: Any
            y: Any
            z: Any

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

        return Foo3

    elif dt is DataType.ATTRS:

        @check_data(**kwargs)
        @attrs.define
        class Foo4:
            x: Any
            y: Any
            z: Any

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

        return Foo5


@pytest.fixture(params=DataType)
def data_type(request: Any) -> DataType:
    x = request.param
    assert isinstance(x, DataType)
    return x


def test_basic(data_type: DataType) -> None:
    FooXYZ = get_datatype(data_type, x="i j", y=["j", "k"], z="j 2")

    FooXYZ(arr(3, 4), arr(4, 3), arr(4, 2))
    FooXYZ(arr(3, 4), arr(4, 3), None)

    with raises_literal("y dim 0: expected j=5 got 7"):
        FooXYZ(arr(3, 5), arr(7, 7), arr(5, 2))

    with raises_literal("z dim 1: expected 2=2 got 3"):
        FooXYZ(arr(1, 2), arr(2, 1), arr(2, 3))


def test_variadic(data_type: DataType) -> None:
    FooProduct = get_datatype(data_type, z="*x  *y", x="*x", y="*y")

    FooProduct(arr(3), arr(7), arr(3, 7))
    FooProduct(arr(3, 5), None, arr(3, 5, 7))

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


def test_bad_data_type() -> None:
    with raises_literal("Unexpected data type", TypeError):

        @check_data(x="i")
        class Foo:
            def __init__(self, x: Any) -> None:
                self.x = x
