from typing import List

from eincheck import (
    check_shapes,
    parser_cache_clear,
    parser_cache_info,
    parser_resize_cache,
)
from eincheck.cache import ResizeableLruCache
from tests.testing_utils import arr

FUNC_CALLS: List[str] = []


def clear_func_calls() -> None:
    global FUNC_CALLS
    FUNC_CALLS.clear()


def foo(x: str) -> str:
    """Docstring."""
    global FUNC_CALLS
    FUNC_CALLS.append(x)
    return x[::-1]


def test_resizeable_cache() -> None:
    cached_foo = ResizeableLruCache(foo)

    clear_func_calls()

    cached_foo("hello")
    cached_foo("world")
    cached_foo("hello")
    cached_foo("foo")

    assert FUNC_CALLS == ["hello", "world", "foo"]
    cache_info = cached_foo.cache_info()
    assert cache_info.currsize == 3
    assert cache_info.hits == 1
    assert cache_info.misses == 3
    assert cache_info.maxsize == 128

    clear_func_calls()
    cached_foo.reset_maxsize(2)
    assert cached_foo.cache_info().currsize == 0

    cached_foo("hello")
    cached_foo("world")
    cached_foo("hello")
    cached_foo("foo")
    cached_foo("bar")
    cached_foo("hello")
    cached_foo("hello")

    assert FUNC_CALLS == ["hello", "world", "foo", "bar", "hello"]
    cache_info = cached_foo.cache_info()
    assert cache_info.currsize == 2
    assert cache_info.hits == 2
    assert cache_info.misses == 5
    assert cache_info.maxsize == 2


def test_parser_cache() -> None:
    parser_cache_clear()

    check_shapes((arr(2, 3, 4), "a b c"), (arr(2, 3), "a b"))
    check_shapes((arr(2, 3, 4), "a b c"), (arr(2, 3, 4, 5), "a b c (c+1)"))

    cache_info = parser_cache_info()
    assert cache_info.currsize == 3
    assert cache_info.hits == 1
    assert cache_info.misses == 3
    assert cache_info.maxsize == 128

    parser_resize_cache(3)

    check_shapes((arr(2, 3, 4), "a b c"), (arr(2, 3), "a b"))
    check_shapes(
        (arr(2), "a"),
        (arr(3), "(a+1)"),
        (arr(2, 3), "a b"),
        (arr(2, 3, 4), "a b c"),
        (arr(2, 3, 4, 5), "a b c (c+1)"),
    )
    check_shapes((arr(2, 3, 4), "a b c"), (arr(2, 3), "a b"))

    cache_info = parser_cache_info()
    assert cache_info.currsize == 3
    assert cache_info.hits == 3
    assert cache_info.misses == 6
    assert cache_info.maxsize == 3

    parser_resize_cache(None)
    check_shapes((arr(2, 3, 4), "a b c"), (arr(2, 3), "a b"))
    check_shapes(
        (arr(2), "a"),
        (arr(3), "(a+1)"),
        (arr(2, 3), "a b"),
        (arr(2, 3, 4), "a b c"),
        (arr(2, 3, 4, 5), "a b c (c+1)"),
    )
    check_shapes((arr(2, 3, 4), "a b c"), (arr(2, 3), "a b"))

    cache_info = parser_cache_info()
    assert cache_info.currsize == 5
    assert cache_info.hits == 4
    assert cache_info.misses == 5
    assert cache_info.maxsize is None


def test_parser_cache_unknown_ndims() -> None:
    check_shapes((arr(2, 3, 4), "a *b"))
