from typing import List, Optional, Tuple

import lark
import pytest

from eincheck.parser.dim_spec import DimSpec, DimType
from eincheck.parser.expressions import AddOp, DataExpr, Literal, MulOp, Variable
from eincheck.parser.grammar import ShapeArg, create_shape_spec
from tests.testing_utils import raises_literal


def _simple_test_case(s: str) -> Tuple[ShapeArg, List[DimSpec]]:
    expected = [DimSpec.create(x) for x in s.split()]
    return s, expected


TEST_CASES = [
    _simple_test_case("i j Hello World i I"),
    _simple_test_case("i 2  1 42 HELLO  ii"),
    ("...", [DimSpec(None, DimType.REPEATED)]),
    (
        "5 ...  k",
        [
            DimSpec.create_literal(5),
            DimSpec(None, DimType.REPEATED),
            DimSpec.create_variable("k"),
        ],
    ),
    (
        "*i j",
        [DimSpec.create_variable("i").make_variadic(), DimSpec.create_variable("j")],
    ),
    (
        "i* j",
        [DimSpec.create_variable("i").make_repeated(), DimSpec.create_variable("j")],
    ),
    ("7* j", [DimSpec.create_literal(7).make_repeated(), DimSpec.create_variable("j")]),
    (
        "*i (j +1)",
        [
            DimSpec.create_variable("i").make_variadic(),
            DimSpec(AddOp(Variable("j"), Literal(1))),
        ],
    ),
    (
        "Hello _ world",
        [
            DimSpec.create_variable("Hello"),
            DimSpec(None),
            DimSpec.create_variable("world"),
        ],
    ),
    ("((2 * i) + 1) ", [DimSpec(AddOp(MulOp(Literal(2), Variable("i")), Literal(1)))]),
    ("(2 * (i + 1)) ", [DimSpec(MulOp(Literal(2), AddOp(Variable("i"), Literal(1))))]),
    ("$", [DimSpec(DataExpr())]),
    (["$"], [DimSpec(DataExpr())]),
    (
        ["i", "2", 3, "_", None],
        [
            DimSpec.create_variable("i"),
            DimSpec.create_literal(2),
            DimSpec.create_literal(3),
            DimSpec(None),
            DimSpec(None),
        ],
    ),
]


@pytest.mark.parametrize("s,expected", TEST_CASES)
def test_parser(s: ShapeArg, expected: List[DimSpec]) -> None:
    got = create_shape_spec(s)
    assert len(got.dims) == len(expected)

    for i, (g, e) in enumerate(zip(got.dims, expected)):
        assert g == e, i

    # ShapeSpec adds square brackets around shape.
    repro_got = create_shape_spec(str(got)[1:-1])

    for i, (g, e) in enumerate(zip(repro_got.dims, expected)):
        assert g == e, i


BAD_ARGS: List[Tuple[ShapeArg, Optional[str]]] = [
    ("i $", None),
    ("... 2+1", None),
    ("i$", None),
    (["i%"], "Variable name should be made of only ascii letters, got i%"),
    (["2.0"], "Variable name should be made of only ascii letters, got 2.0"),
    ([""], "Variable name must not be empty"),
    ([3.2], "Unexpected type: float"),  # type: ignore[list-item]
]


@pytest.mark.parametrize("s,error", BAD_ARGS)
def test_bad_args(s: ShapeArg, error: Optional[str]) -> None:
    if error:
        with raises_literal(error):
            create_shape_spec(s)
    else:
        with pytest.raises(lark.UnexpectedInput):
            create_shape_spec(s)
