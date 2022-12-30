import operator
import random
from typing import Any

import pytest

from eincheck.parser.expressions import (
    BroadcastOp,
    ConcatOp,
    Literal,
    MulOp,
    Variable,
    _ScalarBinaryOp,
)
from tests.utils import raises_literal


@pytest.mark.parametrize("x", [1, 7])
def test_literal(x: int) -> None:
    e = Literal(x)
    assert str(e) == str(x)
    assert e.eval({}) == x


@pytest.mark.parametrize("x", ["i", "HELLO"])
def test_variable(x: str) -> None:
    e = Variable(x)
    assert str(e) == x
    v = random.randint(1, 10)
    assert e.eval({x: v}) == v
    t = (v, 2, v)
    assert e.eval({x: t, "y": 1, "z": v}) == t

    with pytest.raises(KeyError, match=x):
        e.eval({})


@pytest.mark.parametrize(
    "op,func",
    [("+", operator.add), ("*", operator.mul), ("-", operator.sub), ("$$$", max)],
)
def test_binary_op(op: str, func: Any) -> None:
    x = random.randint(1, 7)
    y = random.randint(1, 7)

    e = _ScalarBinaryOp(Literal(x), Variable("y"), op, func)
    assert str(e) == f"({x}{op}y)"
    assert e.eval(dict(y=y)) == func(x, y)


def test_nested() -> None:
    e = MulOp(
        _ScalarBinaryOp(Variable("x"), Literal(1), "add", operator.add), Variable("y")
    )
    assert str(e) == "((xadd1)*y)"
    assert e.eval(dict(x=4, y=2)) == 10

    with pytest.raises(KeyError, match="y"):
        e.eval(dict(x=1))


def test_broadcast() -> None:
    e = BroadcastOp(Variable("x"), Variable("y"))

    assert e.eval(dict(x=(2, 3, 1), y=(2, 1, 4))) == (2, 3, 4)
    assert e.eval(dict(x=(2, 3, 1), y=(4,))) == (2, 3, 4)
    assert e.eval(dict(x=(3, 5), y=(4, 1, 5))) == (4, 3, 5)

    with raises_literal("Shape mismatch in dim 2: 7 vs 3"):
        e.eval(dict(x=(2, 1, 7), y=(3, 3)))


def test_mismatch() -> None:
    e = MulOp(Variable("a"), Variable("b"))

    assert e.eval(dict(a=2, b=3)) == 6

    with raises_literal("Expected int for a in (a*b), got (3, 2)"):
        e.eval(dict(b=5, a=(3, 2)))

    with raises_literal("Expected int for b in (a*b), got (3, 2)"):
        e.eval(dict(a=5, b=(3, 2)))

    e2 = ConcatOp(Variable("i"), Variable("j"))

    assert e2.eval(dict(i=(3, 1), j=(1, 3))) == (3, 1, 1, 3)

    with raises_literal("Expected tuple for i in (i||j), got 2"):
        e2.eval(dict(i=2, j=(3,)))

    with raises_literal("Expected tuple for j in (i||j), got 3"):
        e2.eval(dict(i=(2,), j=3))
