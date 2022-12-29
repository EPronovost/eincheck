from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

from eincheck.parser.dim_spec import DimSpec
from eincheck.types import ShapeVariable


@dataclasses.dataclass
class ShapeSpec:
    dims: List[DimSpec]
    _unknown_n_dims_indices_cache: Optional[Tuple[int, List[int]]] = dataclasses.field(
        default=None
    )

    def __str__(self) -> str:
        return "[" + " ".join(str(d) for d in self.dims) + "]"

    def is_checkable(self, bindings: Dict[str, ShapeVariable]) -> bool:
        if len(self.unknown_n_dims_indices(bindings)) > 1:
            return False

        return all(d.is_checkable(bindings) for d in self.dims)

    def unknown_n_dims_indices(self, bindings: Dict[str, ShapeVariable]) -> List[int]:
        if self._unknown_n_dims_indices_cache and self._unknown_n_dims_indices_cache[
            0
        ] == hash(tuple(bindings.items())):
            return self._unknown_n_dims_indices_cache[1]

        return [i for i, d in enumerate(self.dims) if d.n_dims(bindings) is None]

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
