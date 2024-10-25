from __future__ import annotations

import dataclasses
import enum
from typing import Collection, Dict, Optional, Union

from eincheck.parser.expressions import DataExpr, Expr, Literal, Variable
from eincheck.types import ShapeVariable


@enum.unique
class DimType(enum.Enum):
    # Expression evaluates to a single integer, DimSpec matches a single dimension.
    SINGLE = "single"
    # Expression evaluates to a tuple, DimSpec matches multiple dimensions.
    VARIADIC = "variadic"
    # Expression evaluates to a single integer, DimSpec matches multiple dimensions.
    REPEATED = "repeated"


@dataclasses.dataclass(frozen=True)
class DimSpec:
    """A specification for one part of a shape.

    This can be more than one dimension if matches_multiple is True.
    A full shape spec is made up of a sequence of DimSpecs.
    """

    # Expression for the expected value of this DimSpec.
    # This can evaluate to a single int (if not variadic) or a
    # tuple of ints (if variadic).
    # None represents an unconstrained dimension (i.e. _ or ...).
    value: Optional[Expr]
    type: DimType = dataclasses.field(default=DimType.SINGLE)
    # If true, will accept any shape that can be broadcast to value.
    can_broadcast: bool = dataclasses.field(default=False)

    def __str__(self) -> str:
        s = str(self.value) if self.value else "_"
        if self.type is DimType.VARIADIC:
            s = "*" + s
        elif self.type is DimType.REPEATED:
            s = s + "*"
        if self.can_broadcast:
            s = s + "!"
        return s

    @property
    def matches_multiple(self) -> bool:
        return self.type is not DimType.SINGLE

    def make_variadic(self) -> DimSpec:
        assert self.type is DimType.SINGLE
        return dataclasses.replace(self, type=DimType.VARIADIC)

    def make_repeated(self) -> DimSpec:
        assert self.type is DimType.SINGLE
        return dataclasses.replace(self, type=DimType.REPEATED)

    def make_can_broadcast(self) -> DimSpec:
        return dataclasses.replace(self, can_broadcast=True)

    @staticmethod
    def create_literal(x: int) -> DimSpec:
        return DimSpec(Literal(x))

    @staticmethod
    def create_variable(x: str) -> DimSpec:
        return DimSpec(Variable(x))

    @staticmethod
    def create(x: Union[DimSpec, int, str, None]) -> DimSpec:
        if isinstance(x, DimSpec):
            return x
        elif x == "$":
            return DimSpec(DataExpr())
        elif x == "_":
            return DimSpec(None)
        elif isinstance(x, str) and x.isdigit():
            return DimSpec.create_literal(int(x))
        elif isinstance(x, str):
            return DimSpec.create_variable(x)
        elif isinstance(x, int):
            return DimSpec.create_literal(x)
        elif x is None:
            return DimSpec(None)
        else:
            raise ValueError(f"Unexpected type: {type(x).__name__}")

    def is_defined(self, values: Collection[str]) -> bool:
        """Whether self.value can be evaluated given a set of known variables."""
        if self.value is None:
            return True
        return self.value.is_defined(values)

    def is_checkable(self, values: Collection[str]) -> bool:
        """Whether the DimSpec can be used in a shape check with the given values."""
        return (
            isinstance(self.value, Variable) and not self.can_broadcast
        ) or self.is_defined(values)

    def n_dims(self, bindings: Dict[str, ShapeVariable]) -> Optional[int]:
        """Determine the number of dimensions matched by this DimSpec."""
        if self.type is DimType.SINGLE:
            return 1
        if self.type is DimType.REPEATED:
            return None

        if self.value is not None and self.is_defined(bindings):
            got = self.value.eval(bindings)
            if isinstance(got, tuple):
                return len(got)

        return None
