import re
from typing import Any, Optional, Sequence, Type

import _pytest.python_api
import numpy as np
import numpy.typing as npt
import pytest


def raises_literal(
    x: str, e: Type[BaseException] = ValueError
) -> _pytest.python_api.RaisesContext[BaseException]:
    return pytest.raises(e, match=re.escape(x) + "(?s:.*)")


def arr(*dims: int) -> npt.NDArray[np.float64]:
    """Create a random numpy array with the given dimensions."""
    return np.array(np.random.randn(*dims), dtype=np.float64)


class Dummy:
    def __init__(self, shape: Sequence[Optional[int]], **kwargs: Any):
        self.shape = tuple(shape)
        self.__dict__.update(kwargs)
