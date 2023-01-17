from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Set, Tuple

from eincheck.parser.dim_spec import DimSpec
from eincheck.parser.expressions import DataExpr
from eincheck.types import ShapeVariable


@dataclasses.dataclass
class ShapeSpec:
    dims: List[DimSpec]
    _unknown_n_dims_indices_cache: Optional[Tuple[int, List[int]]] = dataclasses.field(
        default=None
    )

    def __str__(self) -> str:
        return "[" + " ".join(str(d) for d in self.dims) + "]"

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, ShapeSpec) and __o.dims == self.dims

    def is_checkable(self, bindings: Dict[str, ShapeVariable]) -> bool:
        if len(self.unknown_n_dims_indices(bindings)) > 1:
            return False

        return all(d.is_checkable(bindings) for d in self.dims)

    def unknown_n_dims_indices(self, bindings: Dict[str, ShapeVariable]) -> List[int]:
        key = hash(tuple(bindings.items()))
        if (
            self._unknown_n_dims_indices_cache
            and self._unknown_n_dims_indices_cache[0] == key
        ):
            return self._unknown_n_dims_indices_cache[1]

        out = [i for i, d in enumerate(self.dims) if d.n_dims(bindings) is None]
        self._unknown_n_dims_indices_cache = (key, out)
        return out

    def min_rank(self, bindings: Dict[str, ShapeVariable]) -> int:
        n_dims = [d.n_dims(bindings) for d in self.dims]
        return sum(x for x in n_dims if x is not None)

    def matched_indices(
        self, bindings: Dict[str, ShapeVariable], n_dims: int
    ) -> List[Tuple[DimSpec, int, int]]:
        n_dims_per_spec = [d.n_dims(bindings) for d in self.dims]
        min_rank = sum(x for x in n_dims_per_spec if x is not None)
        if min_rank > n_dims:
            raise RuntimeError(f"Expected rank at least {min_rank}, got {n_dims}")

        none_inds = [i for i, x in enumerate(n_dims_per_spec) if x is None]
        if len(none_inds) > 1:
            raise RuntimeError(
                "Cannot determine matched indices: "
                f"multiple DimSpec with unknown n_dims {none_inds}"
            )

        i = 0
        output = []
        for d, x in zip(self.dims, n_dims_per_spec):
            y = (n_dims - min_rank) if x is None else x
            output.append((d, i, i + y))
            i += y

        if i != n_dims:
            raise RuntimeError(f"Expected rank {i}, got {n_dims}")

        return output

    @property
    def is_data_expr(self) -> bool:
        return len(self.dims) == 1 and isinstance(self.dims[0].value, DataExpr)

    @property
    def variables(self) -> Set[str]:
        x = set()
        for d in self.dims:
            if d.value is not None:
                x |= d.value.variables
        return x
