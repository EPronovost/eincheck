import doctest
import os
from typing import NamedTuple

import attrs
import numpy as np
import numpy.typing as npt

from eincheck import check_data, check_func, check_shapes

r = doctest.testfile(
    os.path.join(os.path.dirname(__file__), "README.md"),
    module_relative=False,
    globs=dict(
        attrs=attrs,
        check_data=check_data,
        check_func=check_func,
        check_shapes=check_shapes,
        np=np,
        npt=npt,
        randn=np.random.randn,
        NamedTuple=NamedTuple,
    ),
    optionflags=doctest.DONT_ACCEPT_TRUE_FOR_1 ^ doctest.REPORT_UDIFF,
)

print(f"Doctest: {r.attempted} attempted, {r.attempted - r.failed} passed")
assert r.failed == 0
