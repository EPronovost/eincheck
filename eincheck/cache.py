from functools import _CacheInfo, lru_cache
from typing import Callable, Generic, Optional, TypeVar

from typing_extensions import ParamSpec

_T = TypeVar("_T")
_P = ParamSpec("_P")


class ResizeableLruCache(Generic[_P, _T]):
    def __init__(self, func: Callable[_P, _T], maxsize: Optional[int] = 128):
        self._func = func
        self._cached_func = lru_cache(maxsize=maxsize)(func)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        return self._cached_func(*args, **kwargs)  # type: ignore[arg-type]

    def reset_maxsize(self, maxsize: Optional[int]) -> None:
        self._cached_func = lru_cache(maxsize=maxsize)(self._func)

    def cache_info(self) -> _CacheInfo:
        return self._cached_func.cache_info()

    def cache_clear(self) -> None:
        self._cached_func.cache_clear()


def resizeable_lru_cache(
    maxsize: Optional[int] = 128,
) -> Callable[[Callable[_P, _T]], ResizeableLruCache[_P, _T]]:
    """Resizable version of functools.lru_cache."""
    return lambda f: ResizeableLruCache(f, maxsize=maxsize)
