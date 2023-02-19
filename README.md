# eincheck

[![CI](https://github.com/epronovost/eincheck/actions/workflows/pr.yaml/badge.svg)](https://github.com/epronovost/eincheck/actions/workflows/pr.yaml)

Tensor shape checks inspired by einstein notation

[Read the docs!](https://eincheck.readthedocs.io/en/main/)
[Pypi](https://pypi.org/project/eincheck/)

## Overview

This library has three main functions:

* `check_shapes` takes tuples of `(Tensor, shape)` and checks that all the Tensors match the shapes

```
check_shapes((x, "i 3"), (y, "i 3"))
```

* `check_func` is a function decorator to check the input and output shapes of a function

```
@check_func("*i x, *i y -> *i (x + y)")
def concat(a, b):
    return np.concatenate([a, b], -1)
```

* `check_data` is a class decorator to check the fields of a data class

```
@check_data(start="i 2", end="i 2")
class LineSegment2D(NamedTuple):
    start: torch.Tensor
    end: torch.Tensor
```
