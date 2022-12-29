from typing import List, Tuple

import pytest

from eincheck.parser.dim_spec import DimSpec, DimType
from eincheck.parser.expressions import AddOp, Literal, Variable
from eincheck.parser.grammar import create_shape_spec


def _simple_test_case(s: str) -> Tuple[str, List[DimSpec]]:
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
]


@pytest.mark.parametrize("s,expected", TEST_CASES)
def test_parser(s: str, expected: List[DimSpec]) -> None:
    got = create_shape_spec(s)
    assert len(got.dims) == len(expected)

    for i, (g, e) in enumerate(zip(got.dims, expected)):
        assert g == e, i
