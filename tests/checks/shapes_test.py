import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from eincheck.checks.shapes import (
    _check_dim_spec,
    _is_broadcast_compatible,
    check_shapes,
)
from eincheck.parser.dim_spec import DimSpec
from eincheck.parser.expressions import Variable
from eincheck.parser.grammar import ShapeArg
from eincheck.types import ShapeVariable
from tests.testing_utils import Dummy, raises_literal


@dataclasses.dataclass(frozen=True)
class _TestCase:
    args: List[Tuple[Optional[Sequence[Optional[int]]], ShapeArg]]
    kwargs: Dict[str, Tuple[Optional[Sequence[Optional[int]]], ShapeArg]] = (
        dataclasses.field(default_factory=dict)
    )
    out_bindings: Dict[str, ShapeVariable] = dataclasses.field(default_factory=dict)
    in_bindings: Dict[str, ShapeVariable] = dataclasses.field(default_factory=dict)
    error: str = dataclasses.field(default="")


TEST_CASES = [
    _TestCase([]),
    _TestCase([((3, 4), ["i", 4])], out_bindings={"i": 3}),
    _TestCase([((3, 4), ["i", 4]), ((3, 7), "i 7")], out_bindings={"i": 3}),
    _TestCase([((3, 2), "i (i-1)"), ((3, 7), "i 7")], out_bindings={"i": 3}),
    _TestCase(
        [((3, 5), ["i", 4]), ((3, 7), "i (1+(2*i))")],
        error="arg0 dim 1: expected 4=4 got 5",
    ),
    _TestCase(
        [((3, 4), ["i", 4]), ((4, 7), "i (1+(2*i))")],
        error="arg1 dim 0: expected i=3 got 4",
    ),
    _TestCase(
        [],
        kwargs=dict(foo=((4, 7), "i (1+(2*i))")),
        error="foo dim 1: expected (1+(2*i))=9 got 7",
    ),
    _TestCase([((3, 4), "3 4 5")], error="arg0: expected rank 3, got shape (3, 4)"),
    _TestCase([((3, 3, 3, 4), "i* 4")], out_bindings={"i": 3}),
    _TestCase([((3, 4), "i* (i+1)")], out_bindings={"i": 3}),
    _TestCase([((4,), "7* 4")]),
    _TestCase([((3, 3, 4), "i*")], error="arg0 dim 2: expected i=3 got 4"),
    _TestCase(
        [((3, 4), "i i* _ _")], error="arg0: expected rank at least 3, got shape (3, 4)"
    ),
    _TestCase(
        [], kwargs=dict(x=((5, 5, 6, 5), "i* j i")), out_bindings={"i": 5, "j": 6}
    ),
    _TestCase([((3, 4), "i i* j*")], error="Unable to determine bindings for: arg0"),
    _TestCase([((3, 4), "i 4"), ((3, 7), "i (1+( 2 *i))")], out_bindings={"i": 3}),
    _TestCase(
        [((3, 4, 5), "*x i"), ((3, 4, 6), "*x j"), ((3, 4, 12), "*x (i+(j+1))")],
        out_bindings={"i": 5, "j": 6, "x": (3, 4)},
    ),
    _TestCase(
        [((1, 2, 1), "*x"), ((1, 1, 3), "*y"), ((1, 2, 3), "*(x ^ y)")],
        out_bindings={"x": (1, 2, 1), "y": (1, 1, 3)},
    ),
    _TestCase(
        [
            ((2, 1, 1, 3, 2, 3, 7), "*((x || y) || (x ^y)) 7"),
            ((2, 1), "*x"),
            ((1, 3), "*y"),
        ],
        out_bindings={"x": (2, 1), "y": (1, 3)},
    ),
    _TestCase(
        [],
        kwargs=dict(foo=((1,), "*x *y")),
        error="Unable to determine bindings for: foo",
    ),
    _TestCase(
        [((2, 1), "*i"), ((3, 4), "i j")],
        error="Found variables in both variadic and non-variadic expressions: i",
    ),
    _TestCase(
        [((3, 4), "i j"), ((2, 1), "*i")],
        error="Found variables in both variadic and non-variadic expressions: i",
    ),
    _TestCase(
        [((2, 1), "*i"), ((3, 3), "i*")],
        error="Found variables in both variadic and non-variadic expressions: i",
    ),
    _TestCase(
        [((3, 3), "i*"), ((2, 1), "*i")],
        error="Found variables in both variadic and non-variadic expressions: i",
    ),
    _TestCase(
        [((3, 3), "i *j"), ((2, 1), "*(i || j)")],
        error="Found variables in both variadic and non-variadic expressions: i",
    ),
    _TestCase(
        [((3, 3), "*i j"), ((2, 1), "*(i + j)")],
        error="Found variables in both variadic and non-variadic expressions: j",
    ),
    _TestCase(
        [((3, 3, 4), "(i || j)"), ((3, 3), "*i"), ((4,), "*j")],
        error="Found variables in both variadic and non-variadic expressions: i j",
    ),
    _TestCase(
        [((3, 3, 4), "*(i + j)"), ((3, 3), "*i"), ((4,), "*j")],
        error="Expected int for i in (i+j), got (3, 3)",
    ),
    _TestCase([((), "a*")]),
    _TestCase([((), "a*"), ((7, 7), "a*")], out_bindings={"a": 7}),
    _TestCase([((3,), "i a*")], out_bindings={"i": 3}),
    _TestCase([((3,), "i a*"), ((3, 7, 7), "i a*")], out_bindings={"i": 3, "a": 7}),
    _TestCase([((3, 4), ["i", 4])], in_bindings={"i": 3}, out_bindings={"i": 3}),
    _TestCase(
        [((3, 4), ["i", 4])],
        in_bindings={"i": 2},
        error="arg0 dim 0: expected i=2 got 3",
    ),
    _TestCase(
        [
            ((None, 16, 16), [None, DimSpec.create_variable("i"), "i"]),
            ((1, None, 16, 16), "1 _ i*"),
            ((7, None, None, None, 16), "7 ... i"),
            ((8, None, None, None, 16), "8 _* i"),
            ((9, None, None, None, 16), "9 *_ i"),
        ],
        out_bindings={"i": 16},
    ),
    _TestCase([((), ""), ((), "..."), ((), "*i"), ((), "j*")], out_bindings={"i": ()}),
    _TestCase([((), "_")], error="arg0: expected rank 1, got shape ()"),
    _TestCase(
        [((2, None), "i j")], error="arg0: tried to match (None,) to j, found None"
    ),
    _TestCase(
        [((2, None), "i *j")], error="arg0: tried to match (None,) to *j, found None"
    ),
    _TestCase(
        [((2, None), "i j*")], error="arg0: tried to match (None,) to j*, found None"
    ),
    _TestCase(
        [((2, None), "i (i+1)")],
        error="arg0 dim 1: tried to check (i+1) against (None,), found None",
    ),
    _TestCase(
        [((2, None), "*(i || j)"), ((2,), "*i"), ((3,), "*j")],
        error="arg0 dims (0, 1): tried to check *(i||j) against (2, None), found None",
    ),
    _TestCase(
        [((2, None), "i (i+1)*")],
        error="arg0 dim 1: tried to check (i+1)* against (None,), found None",
    ),
    _TestCase(
        [((2, 3), "i j"), (None, "i j"), ((2, 2), "i j")],
        error="arg2 dim 1: expected j=3 got 2",
    ),
    _TestCase(
        [((2, 3, 4), ["i", "$", "j"])],
        error="arg0: $ should not be present in the shape spec for a Tensor, "
        "got [i $ j]",
    ),
    _TestCase(
        [((2, 3), "i j!"), ((2, 1), "i j!")],
        in_bindings={"j": 3},
        out_bindings={"i": 2, "j": 3},
    ),
    _TestCase(
        [((2, 3), "i *j!"), ((2, 1), "i *j!")],
        in_bindings={"j": (3,)},
        out_bindings={"i": 2, "j": (3,)},
    ),
    _TestCase(
        [((2, 3), "*i!"), ((2, 1), "*i!"), ((1, 3), "*i!"), ((1, 1), "*i!")],
        in_bindings={"i": (2, 3)},
        out_bindings={"i": (2, 3)},
    ),
    _TestCase(
        [((2, 3), "*i"), ((1, 1), "*i!"), ((), "*i!")],
        out_bindings={"i": (2, 3)},
    ),
    _TestCase(
        [((2, 3), "*i"), ((3,), "*i!"), ((), "*i!")],
        out_bindings={"i": (2, 3)},
    ),
    _TestCase(
        [((2, 3), "*i"), ((5,), "*i!")],
        error="arg1 dims (0,): expected can broadcast to i=(2, 3) got (5,)",
    ),
    _TestCase(
        [((2, 3), "*i"), ((1, 4), "*i!")],
        error="arg1 dims (0, 1): expected can broadcast to i=(2, 3) got (1, 4)",
    ),
    _TestCase(
        [((5, 2, 3), "i *j"), ((5,), "i *j!"), ((5, 3), "i *j!"), ((5, 1), "i *j!")],
        out_bindings={"i": 5, "j": (2, 3)},
    ),
    _TestCase(
        [((5, 2, 3), "i *j"), ((5, 5), "i *j!")],
        error="arg1 dims (1,): expected can broadcast to j=(2, 3) got (5,)",
    ),
    _TestCase(
        [((3,), "*j"), ((1, 3), "*j!")],
        error="arg1 dims (0, 1): expected can broadcast to j=(3,) got (1, 3)",
    ),
    _TestCase(
        [((2, 4), "i (j+1)!"), ((2, 1), "i (j+1)!")],
        in_bindings={"j": 3},
        out_bindings={"i": 2, "j": 3},
    ),
    _TestCase(
        [((2, 3), "i i!")], error="arg0 dim 1: expected can broadcast to i=2 got 3"
    ),
    _TestCase(
        [((3, 4), "*n!")], error="Unable to check: [arg0] missing variables: [n]"
    ),
    _TestCase(
        [((3, 4), "3 n!")], error="Unable to check: [arg0] missing variables: [n]"
    ),
    _TestCase([((), ".")]),
    _TestCase([((3,), ".")], error="arg0: expected rank 0, got shape (3,)"),
    _TestCase(
        [((3,), "*5")],
        error="arg0: expected variadic DimSpec *5 to evaluate to a tuple, got 5",
    ),
]


def test_is_broadcast_compatible() -> None:
    assert _is_broadcast_compatible((), (2, 3))
    assert _is_broadcast_compatible((3,), (2, 3))
    assert _is_broadcast_compatible((1,), (2, 3))
    assert _is_broadcast_compatible((2, 3), (2, 3))
    assert _is_broadcast_compatible((1, 1), (2, 3))
    assert not _is_broadcast_compatible((5,), (2, 3))
    assert not _is_broadcast_compatible((1, 4), (2, 3))
    assert not _is_broadcast_compatible((1, 2, 3), (2, 3))


@pytest.mark.parametrize("case", TEST_CASES)
def test_simple(case: _TestCase) -> None:
    args = [(None if x is None else Dummy(x), y) for x, y in case.args]
    kwargs = {
        k: (None if x is None else Dummy(x), y) for k, (x, y) in case.kwargs.items()
    }
    if case.error:
        with raises_literal(case.error):
            check_shapes(*args, **kwargs, **case.in_bindings)
    else:
        got = check_shapes(*args, **kwargs, **case.in_bindings)
        assert got == case.out_bindings


def test_unexpected_kwarg() -> None:
    with raises_literal("Unexpected kwarg hello"):
        check_shapes(x="hello")  # type: ignore[arg-type]


def test_check_dim_spec_value_none() -> None:
    _check_dim_spec(
        got_shape=(3,), d=DimSpec(None), bindings={}, name="t", msg="", start_idx=0
    )


def test_check_dim_spec_non_variadic_tuple() -> None:
    d = DimSpec(Variable("x"))
    with raises_literal(
        "expected non-variadic DimSpec x to evaluate to an integer, got (1, 2)"
    ):
        _check_dim_spec(
            got_shape=(3,),
            d=d,
            bindings={"x": (1, 2)},
            name="t",
            msg="",
            start_idx=0,
        )
