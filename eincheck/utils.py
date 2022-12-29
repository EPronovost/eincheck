from typing import Optional, Sequence, Tuple, overload

from typing_extensions import Literal

from eincheck.types import Tensor


@overload
def get_shape(x: Tensor, strict: Literal[True]) -> Tuple[Optional[int], ...]:
    ...


@overload
def get_shape(x: Tensor, strict: bool = ...) -> Optional[Tuple[Optional[int], ...]]:
    ...


def get_shape(x: Tensor, strict: bool = False) -> Optional[Tuple[Optional[int], ...]]:
    try:
        if hasattr(x, "shape"):
            s: Optional[Sequence[Optional[int]]] = x.shape  # type: ignore[attr-defined]
            if s is not None:
                return tuple(i if isinstance(i, int) else None for i in s)

        elif isinstance(x, (tuple, list)):
            return (len(x),)

    except Exception:
        if strict:
            raise

    return None
