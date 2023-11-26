from typing import Any

import pytest

from eincheck.checks.func import check_func
from eincheck.checks.shapes import check_shapes
from eincheck.contexts import disable_checks, enable_checks
from tests.utils import arr


def test_disable_check_shapes() -> None:
    with pytest.raises(ValueError):
        check_shapes((arr(7), "a b"))

    with disable_checks():
        check_shapes((arr(7), "a b"))

    with pytest.raises(ValueError):
        with disable_checks():
            with enable_checks():
                check_shapes((arr(7), "a b"))


def test_disable_check_func() -> None:
    @check_func("*x, *x -> *x")
    def foo(x: Any, y: Any) -> Any:
        return x + y

    with pytest.raises(ValueError):
        foo(arr(2, 3), arr(3))

    with disable_checks():
        foo(arr(2, 3), arr(3))

    with pytest.raises(ValueError):
        with disable_checks():
            with enable_checks():
                foo(arr(2, 3), arr(3))
