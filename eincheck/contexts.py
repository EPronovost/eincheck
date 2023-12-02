from contextlib import contextmanager
from typing import ContextManager, Generator

__EINCHECK_ENABLE_CHECKS: bool = True


def _should_do_checks() -> bool:
    return __EINCHECK_ENABLE_CHECKS


@contextmanager
def _set_enable_checks(value: bool) -> Generator[None, None, None]:
    global __EINCHECK_ENABLE_CHECKS
    prev = __EINCHECK_ENABLE_CHECKS
    __EINCHECK_ENABLE_CHECKS = value
    yield
    assert __EINCHECK_ENABLE_CHECKS == value
    __EINCHECK_ENABLE_CHECKS = prev


def enable_checks() -> ContextManager[None]:
    """Enable eincheck to do shape checks."""
    return _set_enable_checks(True)


def disable_checks() -> ContextManager[None]:
    """Disable eincheck from doing shape checks."""
    return _set_enable_checks(False)
