import functools
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Mapping, Optional, Set, TypeVar

from eincheck.checks.shapes import check_shapes
from eincheck.parser.grammar import ShapeArg, create_shape_spec
from eincheck.parser.shape_spec import ShapeSpec

_T = TypeVar("_T")


class DataWrapper(ABC):
    module_name: Optional[str] = None

    @classmethod
    def can_load(cls) -> bool:
        return cls.module_name is None or cls.module_name in sys.modules

    @abstractmethod
    def is_match(self, x: Any) -> bool:
        """Whether x is a data object of the right type."""

    @abstractmethod
    def wrap(self, cls: _T, shapes: Mapping[str, ShapeSpec]) -> _T:
        pass

    @staticmethod
    def check_fields(shapes: Mapping[str, ShapeSpec], got: Set[str]) -> None:
        extra_names = set(shapes) - got
        if extra_names:
            raise ValueError("No field found: [" + " ".join(sorted(extra_names)) + "]")


class NamedTupleWrapper(DataWrapper):
    def is_match(self, x: Any) -> bool:
        return issubclass(x, tuple) and hasattr(x, "_fields")

    def wrap(self, cls: _T, shapes: Mapping[str, ShapeSpec]) -> _T:
        self.check_fields(shapes, set(cls._fields))  # type: ignore[attr-defined]

        _new = cls.__new__

        @functools.wraps(_new)  # type: ignore[misc]
        @classmethod
        def new_new(cls: Any, *a: Any, **k: Any) -> Any:
            out = _new(*a, **k)

            check_shapes(**{k: (getattr(out, k), s) for k, s in shapes.items()})
            return out

        cls.__new__ = new_new  # type: ignore[assignment]

        return cls


def _func_with_check(
    cls: Any, func: str, shapes: Mapping[str, ShapeSpec], append: bool
) -> None:
    old_f = getattr(cls, func)

    if append:

        def new_f(self: Any, *a: Any, **k: Any) -> Any:
            old_f(self, *a, **k)
            check_shapes(**{k: (getattr(self, k), s) for k, s in shapes.items()})

    else:

        def new_f(self: Any, *a: Any, **k: Any) -> Any:
            check_shapes(**{k: (getattr(self, k), s) for k, s in shapes.items()})
            old_f(self, *a, **k)

    new_f = functools.wraps(old_f)(new_f)

    setattr(cls, func, new_f)


class DataclassWrapper(DataWrapper):
    if sys.version_info[:2] < (3, 8):
        module_name = "dataclasses"

    def __init__(self) -> None:
        super().__init__()
        import dataclasses

        self.dataclasses = dataclasses

    def is_match(self, x: Any) -> bool:
        return self.dataclasses.is_dataclass(x)

    def wrap(self, cls: _T, shapes: Mapping[str, ShapeSpec]) -> _T:
        self.check_fields(shapes, {f.name for f in self.dataclasses.fields(cls)})

        if hasattr(cls, "__post_init__"):
            _func_with_check(cls, "__post_init__", shapes, False)
        else:
            _func_with_check(cls, "__init__", shapes, True)
        return cls


class AttrsWrapper(DataWrapper):
    module_name = "attrs"

    def __init__(self) -> None:
        super().__init__()
        import attrs

        self.attrs = attrs

    def is_match(self, x: Any) -> bool:
        return self.attrs.has(x)

    def wrap(self, cls: _T, shapes: Mapping[str, ShapeSpec]) -> _T:
        self.check_fields(
            shapes, {a.name for a in self.attrs.fields(cls)}  # type: ignore[arg-type]
        )

        if hasattr(cls, "__attrs_post_init__"):
            _func_with_check(cls, "__attrs_post_init__", shapes, False)

        else:
            _func_with_check(cls, "__init__", shapes, True)

        return cls


_wrappers: List[DataWrapper] = []

_T_Data = TypeVar("_T_Data")


def check_data(**kwargs: ShapeArg) -> Callable[[_T_Data], _T_Data]:
    shapes = {k: create_shape_spec(v) for k, v in kwargs.items()}

    def wrapper(cls: _T_Data) -> _T_Data:
        for w in _wrappers:
            if w.is_match(cls):
                return w.wrap(cls, shapes)

        for w_cls in DataWrapper.__subclasses__():
            if w_cls.can_load():
                _wrappers.append(w_cls())  # type: ignore[abstract]
                if _wrappers[-1].is_match(cls):
                    return _wrappers[-1].wrap(cls, shapes)

        raise TypeError(f"Unexpected data type {cls}")

    return wrapper
