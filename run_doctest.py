import doctest
import os

import numpy as np

from eincheck import check_data, check_func, check_shapes

r = doctest.testfile(
    os.path.join(os.path.dirname(__file__), "README.md"),
    module_relative=False,
    globs=dict(
        np=np,
        randn=np.random.randn,
        check_data=check_data,
        check_func=check_func,
        check_shapes=check_shapes,
    ),
    optionflags=doctest.DONT_ACCEPT_TRUE_FOR_1 ^ doctest.REPORT_UDIFF,
)

print(f"Doctest: {r.attempted} attempted, {r.attempted - r.failed} passed")
assert r.failed == 0
