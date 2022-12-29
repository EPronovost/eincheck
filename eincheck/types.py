from typing import Optional, Sequence, Tuple, TypeVar, Union

_ShapeType = TypeVar("_ShapeType", bound=Optional[Sequence[Optional[int]]])


Tensor = TypeVar("Tensor")
ShapeVariable = Union[int, Tuple[int, ...]]
