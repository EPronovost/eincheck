import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, Collection, Dict, Generic, Tuple, TypeVar

from eincheck.types import ShapeVariable


class Expr(ABC):
    """Base expression."""

    @abstractmethod
    def __str__(self) -> str:
        """Representation of the expression."""

    @abstractmethod
    def eval(self, values: Dict[str, ShapeVariable]) -> ShapeVariable:
        """Evaluate the expression given existing values."""

    @abstractmethod
    def is_defined(self, values: Collection[str]) -> bool:
        """Whether all the necessary variables are defined."""


class Literal(Expr):
    """A literal value."""

    def __init__(self, x: int):
        self.x = x

    def __str__(self) -> str:
        return str(self.x)

    def eval(self, values: Dict[str, ShapeVariable]) -> ShapeVariable:
        return self.x

    def is_defined(self, values: Collection[str]) -> bool:
        return True

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Literal) and o.x == self.x


class Variable(Expr):
    def __init__(self, x: str):
        self.x = x

    def __str__(self) -> str:
        return self.x

    def eval(self, values: Dict[str, ShapeVariable]) -> ShapeVariable:
        return values[self.x]

    def is_defined(self, values: Collection[str]) -> bool:
        return self.x in values

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
            raise ValueError(f"Expected {self.expected_type} for {self.x}, got {x}")
        if not isinstance(y, self.expected_type):
            raise ValueError(f"Expected {self.expected_type} for {self.y}, got {y}")

        return self.func(x, y)

    def is_defined(self, values: Collection[str]) -> bool:
        return self.x.is_defined(values) and self.y.is_defined(values)


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
