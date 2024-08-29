from typing import NamedTuple

import pytest

from eincheck.utils import get_object, parse_dot_name


class Foo(NamedTuple):
    x: int
    y: int


class Bar(NamedTuple):
    foo: Foo
    z: int


def test_get_object() -> None:
    data = {
        "a": Bar(foo=Foo(x=1, y=2), z=3),
        "b": Foo(x=4, y=5),
        "c": {"list": ["i", "j"], "tuple": ("hello", "world"), "foo": Foo(x=7, y=8)},
    }

    assert get_object("a", data) == data["a"]
    assert get_object("b", data) == data["b"]
    assert get_object("c", data) == data["c"]

    assert get_object("a.foo", data) == Foo(x=1, y=2)
    assert get_object("a.0", data) == Foo(x=1, y=2)
    assert get_object("a.foo.x", data) == 1
    assert get_object("a.foo.y", data) == 2
    assert get_object("a.z", data) == 3
    assert get_object("a.1", data) == 3
    assert get_object("b.x", data) == 4
    assert get_object("b.y", data) == 5
    assert get_object("c.list", data) == ["i", "j"]
    assert get_object("c.list.0", data) == "i"
    assert get_object("c.list.1", data) == "j"
    assert get_object("c.tuple", data) == ("hello", "world")
    assert get_object("c.tuple.0", data) == "hello"
    assert get_object("c.tuple.1", data) == "world"
    assert get_object("c.foo", data) == Foo(x=7, y=8)
    assert get_object("c.foo.x", data) == 7
    assert get_object("c.foo.y", data) == 8
    assert get_object("c.foo.0", data) == 7
    assert get_object("c.foo.1", data) == 8

    assert get_object("x", None) is None
    assert get_object("c.foo.1", None) is None
    assert get_object("0.x", [Foo(1, 2), None]) == 1
    assert get_object("1.x", [Foo(1, 2), None]) is None

    with pytest.raises(KeyError, match="'x'"):
        get_object("x", data)

    with pytest.raises(AttributeError, match="'Bar' object has no attribute 'a'"):
        get_object("a.a", data)

    with pytest.raises(AttributeError, match="'list' object has no attribute 'a'"):
        get_object("c.list.a", data)

    with pytest.raises(IndexError, match="tuple index out of range"):
        get_object("a.2", data)

    with pytest.raises(KeyError, match="'hello'"):
        get_object("c.hello", data)


def test_parse_dot_name() -> None:
    assert parse_dot_name("foo") == ("foo", [])
    assert parse_dot_name("foo.x") == ("foo", ["x"])
    assert parse_dot_name("foo.x.y") == ("foo", ["x", "y"])
