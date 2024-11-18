from typing import Any, Sequence, Tuple, Union

from lark.lark import Lark
from lark.visitors import Transformer

from eincheck.cache import resizeable_lru_cache
from eincheck.parser.dim_spec import DimSpec, DimType
from eincheck.parser.expressions import (
    AddOp,
    BroadcastOp,
    ConcatOp,
    DataExpr,
    Expr,
    MulOp,
    SubOp,
)
from eincheck.parser.shape_spec import ShapeSpec

ShapeArg = Union[str, ShapeSpec, Sequence[Union[DimSpec, str, int, None]]]

grammar = r"""

shape : can_broadcast_dim? (" "+ can_broadcast_dim)*
      | "$" -> dollar
      | "." -> scalar

?expr : value_expr
      | "_" -> underscore

?value_expr : INT   -> int
            | NAME  -> word
            | math

?can_broadcast_dim : dim "!" -> can_broadcast
                   | dim

?dim : expr "*" -> repeated
     | "*" expr -> variadic
     | expr
     | "..." -> ellipse

?math : "(" " "* value_expr " "* "+" " "* value_expr " "* ")" -> add
      | "(" " "* value_expr " "* "-" " "* value_expr " "* ")" -> sub
      | "(" " "* value_expr " "* "*" " "* value_expr " "* ")" -> mul
      | "(" " "* value_expr " "* "||" " "* value_expr " "* ")" -> concat
      | "(" " "* value_expr " "* "^" " "* value_expr " "* ")" -> broadcast
      | "(" " "* value_expr " "* ")"

%import common.INT
%import python.NAME
"""

_parser = Lark(grammar, start="shape")


class TreeToSpec(Transformer):  # type: ignore[type-arg]
    def int(self, s: Any) -> DimSpec:
        (x,) = s
        return DimSpec.create_literal(int(x))

    def word(self, s: Any) -> DimSpec:
        (x,) = s
        return DimSpec.create_variable(x)

    def add(self, s: Any) -> DimSpec:
        x, y = s
        return DimSpec(AddOp(self._get_expr(x), self._get_expr(y)))

    def mul(self, s: Any) -> DimSpec:
        x, y = s
        return DimSpec(MulOp(self._get_expr(x), self._get_expr(y)))

    def sub(self, s: Any) -> DimSpec:
        x, y = s
        return DimSpec(SubOp(self._get_expr(x), self._get_expr(y)))

    def concat(self, s: Any) -> DimSpec:
        x, y = s
        return DimSpec(ConcatOp(self._get_expr(x), self._get_expr(y)))

    def broadcast(self, s: Any) -> DimSpec:
        x, y = s
        return DimSpec(BroadcastOp(self._get_expr(x), self._get_expr(y)))

    def underscore(self, s: Any) -> DimSpec:
        return DimSpec(None)

    def repeated(self, s: Tuple[DimSpec]) -> DimSpec:
        (x,) = s
        assert isinstance(x, DimSpec)
        return x.make_repeated()

    def variadic(self, s: Tuple[DimSpec]) -> DimSpec:
        (x,) = s
        assert isinstance(x, DimSpec)
        return x.make_variadic()

    def can_broadcast(self, s: Tuple[DimSpec]) -> DimSpec:
        (x,) = s
        assert isinstance(x, DimSpec)
        return x.make_can_broadcast()

    def ellipse(self, s: Any) -> DimSpec:
        return DimSpec(None, DimType.REPEATED)

    def shape(self, dims: Sequence[DimSpec]) -> ShapeSpec:
        return ShapeSpec(list(dims))

    def dollar(self, s: Any) -> ShapeSpec:
        return ShapeSpec([DimSpec(DataExpr())])

    def scalar(self, s: Any) -> ShapeSpec:
        return ShapeSpec([])

    @staticmethod
    def _get_expr(x: Any) -> Expr:
        assert isinstance(x, DimSpec)
        assert not x.matches_multiple
        assert x.value is not None
        return x.value


_transformer = TreeToSpec()


@resizeable_lru_cache()
def parse_shape_spec(s: str) -> ShapeSpec:
    """Parse a string into a ShapeSpec."""
    out = _transformer.transform(_parser.parse(s.strip(" ")))
    assert isinstance(out, ShapeSpec)
    return out


def create_shape_spec(arg: ShapeArg) -> ShapeSpec:
    """Convert a generic ShapeArg into a ShapeSpec."""
    if isinstance(arg, ShapeSpec):
        return arg
    if isinstance(arg, str):
        return parse_shape_spec(arg)

    dims = [DimSpec.create(x) for x in arg]
    return ShapeSpec(dims)
