from typing import Optional, Sequence, Tuple

from eincheck.types import Tensor


def get_shape(x: Tensor) -> Optional[Tuple[Optional[int], ...]]:
    if hasattr(x, "shape"):
        s: Optional[Sequence[Optional[int]]] = x.shape
        if s is not None:
            return tuple(i if isinstance(i, int) else None for i in s)

    elif isinstance(x, (tuple, list)):
        return (len(x),)

    return None
