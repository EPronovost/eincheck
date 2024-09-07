from typing import Any, List, Optional, Sequence, Tuple

from eincheck.types import Tensor


def get_shape(x: Tensor) -> Optional[Tuple[Optional[int], ...]]:
    if hasattr(x, "shape"):
        s: Optional[Sequence[Optional[int]]] = x.shape
        if s is not None:
            return tuple(i if isinstance(i, int) else None for i in s)

    elif isinstance(x, (tuple, list)):
        return (len(x),)

    return None


def parse_dot_name(dot_name: str) -> Tuple[str, List[str]]:
    """Split a dot-separated name path into pieces."""
    first_name, *name_path = dot_name.split(".")
    return first_name, name_path


def get_object(dot_name: str, data: Any) -> Any:
    """Use a dot name path to extract an object from data."""
    if not dot_name:
        return data

    first_name, name_path = parse_dot_name(dot_name)

    x = _get_field(data, first_name)
    for n in name_path:
        x = _get_field(x, n)

    return x


def _get_field(obj: Any, field: str) -> Any:
    if obj is None:
        return None

    if field.isdecimal():
        idx = int(field)
        return obj[idx]

    if isinstance(obj, dict):
        return obj[field]

    return getattr(obj, field)
