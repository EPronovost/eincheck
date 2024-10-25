import operator
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Collection, Dict, Generic, Set, Tuple, TypeVar

from eincheck.types import ShapeVariable


class Expr(ABC):
    """Base expression."""

    @abstractmethod
    def __str__(self) -> str:
        """Representation of the expression."""

    @abstractmethod
    def eval(self, values: Dict[str, ShapeVariable]) -> ShapeVariable:
        """Evaluate the expression given existing values."""

    @property
    @abstractmethod
    def variables(self) -> Set[str]:
        """The set of variables used for this expression."""

    def is_defined(self, values: Collection[str]) -> bool:
        """Whether all the necessary variables are defined."""
        return self.variables <= set(values)


class Literal(Expr):
    """A literal value."""

    def __init__(self, x: int):
        self.x = x

    def __str__(self) -> str:
        return str(self.x)

    def eval(self, values: Dict[str, ShapeVariable]) -> ShapeVariable:
        return self.x

    @property
    def variables(self) -> Set[str]:
        return set()

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Literal) and o.x == self.x


class Variable(Expr):
    NAME_REGEX = re.compile(r"[^\W\d]\w*")

    def __init__(self, x: str):
        if not x:
            raise ValueError("Variable name must not be empty")
        if not self.NAME_REGEX.fullmatch(x):
            raise ValueError(f"Variable name should be a valid python name, got {x}")
        self.x = str(x)

    def __str__(self) -> str:
        return self.x

    def eval(self, values: Dict[str, ShapeVariable]) -> ShapeVariable:
        return values[self.x]

    @property
    def variables(self) -> Set[str]:
        return {self.x}

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Variable) and o.x == self.x


_T = TypeVar("_T", int, Tuple[int, ...])


class _BinaryOp(Expr, Generic[_T]):
    expected_type: Any

    def __init__(self, x: Expr, y: Expr, op: str, func: Callable[[_T, _T], _T]):
        self.x = x
        self.y = y
        self.op = op
        self.func: Callable[[_T, _T], _T] = func

    def __str__(self) -> str:
        return f"({self.x}{self.op}{self.y})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, type(self))
            and o.x == self.x
            and o.y == self.y
            and o.op == self.op
        )

    def eval(self, values: Dict[str, ShapeVariable]) -> _T:
        x = self.x.eval(values)
        y = self.y.eval(values)
        if not isinstance(x, self.expected_type):
            raise ValueError(
                f"Expected {self.expected_type.__name__} for {self.x} in {self}, "
                f"got {x}"
            )
        if not isinstance(y, self.expected_type):
            raise ValueError(
                f"Expected {self.expected_type.__name__} for {self.y} in {self}, "
                f"got {y}"
            )

        return self.func(x, y)

    @property
    def variables(self) -> Set[str]:
        return self.x.variables | self.y.variables


class _ScalarBinaryOp(_BinaryOp[int]):
    expected_type = int


class AddOp(_ScalarBinaryOp):
    def __init__(self, x: Expr, y: Expr):
        super().__init__(x, y, "+", operator.add)


class MulOp(_ScalarBinaryOp):
    def __init__(self, x: Expr, y: Expr):
        super().__init__(x, y, "*", operator.mul)


class SubOp(_ScalarBinaryOp):
    def __init__(self, x: Expr, y: Expr):
        super().__init__(x, y, "-", operator.sub)


class _TupleBinaryOp(_BinaryOp[Tuple[int, ...]]):
    expected_type = tuple


class ConcatOp(_TupleBinaryOp):
    def __init__(self, x: Expr, y: Expr):
        super().__init__(x, y, "||", operator.add)


class BroadcastOp(_TupleBinaryOp):
    def __init__(self, x: Expr, y: Expr):
        super().__init__(x, y, "^", BroadcastOp.broadcast)

    @staticmethod
    def broadcast(x: Tuple[int, ...], y: Tuple[int, ...]) -> Tuple[int, ...]:
        len_diff = len(x) - len(y)
        if len_diff > 0:
            y = (1,) * len_diff + y
        elif len_diff < 0:
            x = (1,) * (-len_diff) + x

        assert len(x) == len(y)

        out = []
        for idx, (xx, yy) in enumerate(zip(x, y)):
            if xx == 1:
                out.append(yy)
            elif yy == 1:
                out.append(xx)
            elif xx != yy:
                raise ValueError(f"Shape mismatch in dim {idx}: {xx} vs {yy}")
            else:
                out.append(xx)

        return tuple(out)


class DataExpr(Expr):
    def __str__(self) -> str:
        return "$"

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, DataExpr)

    @property
    def variables(self) -> Set[str]:
        return set()

    def eval(self, values: Dict[str, ShapeVariable]) -> ShapeVariable:
        raise RuntimeError("Tried to evaluate DataExpr")
