# eincheck

[![CI](https://github.com/epronovost/eincheck/actions/workflows/pr.yaml/badge.svg)](https://github.com/epronovost/eincheck/actions/workflows/pr.yaml)

Easy Tensor shape checks

## Why check shapes?

Common Tensor data types don't explicitly document the data shapes in the code.
Shape checks and comments both add richer shape information, making Tensor manipulating code much easier to read.
While comments are better than nothing, explicit shape checks are enforced to make sure the shape documentation is correct.

## Overview

This library has three main functions:

* `check_shapes` takes tuples of `(Tensor, shape)` and checks that all the Tensors match the shapes

* `check_func` is a function decorator to check the input and output shapes of a function

* `check_data` is a class decorator to check the fields of a data class

## Eincheck by Example

### Check Shapes
```
>>> from eincheck import check_shapes
>>> from numpy.random import randn

```
A shape specification defines a set of constraints that a Tensor's shape must satisfy.
A shape spec is made up of dimension specifications separated by spaces.
The simplest dimension specification is a variable (e.g. `i`).

If the checks pass, check_shapes returns the value of variables.
```
>>> check_shapes(
...     (randn(3, 4, 5), "i j k"),
...     (randn(5, 3), "k i"),
... )
{'i': 3, 'j': 4, 'k': 5}

```

If the checks fail, an exception is raised with information to help debug.
```
>>> check_shapes(
...     (randn(3, 4, 5), "i j k"),
...     (randn(5, 4), "k i"),
... )
Traceback (most recent call last):
    ...
ValueError: arg1 dim 1: expected i=3 got 4
    i=3
    j=4
    k=5
  arg0: got (3, 4, 5) expected [i j k]
  arg1: got (5, 4)    expected [k i]

```

Keyword arguments can be used to give the Tensors different names in the error message.
If the shape checks pass, there's no difference between positional and keyword arguments.
```
>>> check_shapes(
...     foo=(randn(3, 4, 5), "i j k"),
...     bar=(randn(5, 4), "k i"),
... )
Traceback (most recent call last):
  ...
ValueError: bar dim 1: expected i=3 got 4
    i=3
    j=4
    k=5
  foo: got (3, 4, 5) expected [i j k]
  bar: got (5, 4)    expected [k i]

```

Keyword arguments can also specify the value of variables.
```
>>> check_shapes(foo=(randn(7, 8, 9), "a b c"), b=8)
{'b': 8, 'a': 7, 'c': 9}

>>> check_shapes(foo=(randn(7, 8, 9), "a b c"), b=2)
Traceback (most recent call last):
  ...
ValueError: foo dim 1: expected b=2 got 8
    b=2
    a=7
    c=9
  foo: got (7, 8, 9) expected [a b c]

```
This can be used to chain shape checks.

```
>>> x = randn(3, 4)
>>> y = randn(4, 5)
>>> shape_vars = check_shapes(x=(x, "i j"), y=(y, "j k"))
>>> check_shapes((x @ y, "i k"), **shape_vars)
{'i': 3, 'j': 4, 'k': 5}
>>> check_shapes((randn(2, 5), "i k"), **shape_vars)
Traceback (most recent call last):
    ...
ValueError: arg0 dim 0: expected i=3 got 2
    i=3
    j=4
    k=5
  arg0: got (2, 5) expected [i k]

```

### Complex Dimension Specs

Shapes can be more than just individual characters.
A variable can be any sequence of `[a-zA-Z]+`.
```
>>> check_shapes((randn(1, 2, 3), "first second third"))
{'first': 1, 'second': 2, 'third': 3}

```

The specification for a dimension can be a literal integer.
```
>>> check_shapes((randn(4, 5), "x 5"))
{'x': 4}

>>> check_shapes((randn(4, 5), "x 4"))
Traceback (most recent call last):
  ...
ValueError: arg0 dim 1: expected 4=4 got 5
    x=4
  arg0: got (4, 5) expected [x 4]

```

In fact, a dimension spec can be an expression using variables and integers enclosed in parentheses.
There are three binary operators for integers: `+`, `-`, and `*`.
```
>>> x = randn(3, 5)
>>> y = randn(3, 2)
>>> check_shapes(
...     x=(x, "i j"),
...     y=(y, "i k"),
...     z=(np.concatenate([x, y], -1), "i (j + k)"),
... )
{'i': 3, 'j': 5, 'k': 2}

>>> check_shapes(
...     a=(randn(1, 2), "i j"),
...     b=(randn(3, 4), "x y"),
...     c=(randn(7, 1), "(i + (j * x)) ((y - x) * i)"),
... )
{'i': 1, 'j': 2, 'x': 3, 'y': 4}

>>> check_shapes(
...     a=(randn(1, 2), "i j"),
...     b=(randn(3, 4), "x y"),
...     c=(randn(7, 2), "(i + (j * x)) ((y - x) * i)"),
... )
Traceback (most recent call last):
  ...
ValueError: c dim 1: expected ((y-x)*i)=1 got 2
    i=1
    j=2
    x=3
    y=4
  a: got (1, 2) expected [i j]
  b: got (3, 4) expected [x y]
  c: got (7, 2) expected [(i+(j*x)) ((y-x)*i)]

```

Eincheck is not a general equation solver; the value of a variable can only be determined from a dimension that has just that variable (i.e. not an expression).
If the value of a variable cannot be determined, an exception will be raised.
```
>>> check_shapes((randn(42), "(i * 2)"), i=21)
{'i': 21}
>>> check_shapes((randn(42), "(i * 2)"))
Traceback (most recent call last):
  ...
ValueError: Unable to check: [arg0] missing variables: [i]
  arg0: got (42,) expected [(i*2)]

```

An underscore denotes a dimension that can take any value.
```
>>> check_shapes((randn(7, 8, 9), "i _ j"))
{'i': 7, 'j': 9}

>>> check_shapes((randn(7, 9), "i _ j"))
Traceback (most recent call last):
  ...
ValueError: arg0: expected rank 3, got shape (7, 9)
  arg0: got (7, 9) expected [i _ j]

```

### Repeated Dimensions

There are two ways a single dimension spec can match multiple dimensions in the Tensor shape.
The first is as a repeated dimension.
Adding a `*` at the end of a dimension expression will match 0 or more dimensions that match the expression.

```
>>> check_shapes((randn(4, 1, 1, 1, 2), "i 1* j"))
{'i': 4, 'j': 2}
>>> check_shapes((randn(4, 1, 2), "i 1* j"))
{'i': 4, 'j': 2}
>>> check_shapes((randn(4, 2), "i 1* j"))
{'i': 4, 'j': 2}
>>> check_shapes((randn(4, 2, 2), "i 1* j"))
Traceback (most recent call last):
  ...
ValueError: arg0 dim 1: expected 1=1 got 2
    i=4
    j=2
  arg0: got (4, 2, 2) expected [i 1* j]
>>> check_shapes((randn(3, 3, 6), "i* (2 * i)"))
{'i': 3}
>>> check_shapes((randn(3, 6, 6), "i (2 * i)*"))
{'i': 3}

```

### Variadic Dimensions

The second way a single dimension spec can match multiple dimensions are a variadic dimension spec.
This is denoted by adding a `*` in front of the dimension spec.

Similar to `*args` in a Python function, a variadic dimension spec will match a tuple of zero or more dimensions in the Tensor shape.

```
>>> check_shapes((randn(7, 8, 9), "*i j"))
{'i': (7, 8), 'j': 9}
>>> check_shapes(
...     foo=(randn(3, 4, 5), "*i j"),
...     bar=(randn(3, 3, 3), "*i k"),
... )
Traceback (most recent call last):
  ...
ValueError: bar dims (0, 1): expected i=(3, 4) got (3, 3)
    i=(3, 4)
    j=5
    k=3
  foo: got (3, 4, 5) expected [*i j]
  bar: got (3, 3, 3) expected [*i k]

```

Variadic expressions should not evaluate to an integer.

```
>>> check_shapes((randn(2, 3), "*2"))
Traceback (most recent call last):
  ...
ValueError: arg0: expected variadic DimSpec *2 to evaluate to a tuple, got 2
  arg0: got (2, 3) expected [*2]
>>> check_shapes(
...     x0=(randn(2, 3), "i j"),
...     x1=(randn(4, 5, 6), "*(i + j)"),
... )
Traceback (most recent call last):
  ...
ValueError: Found variables in both variadic and non-variadic expressions: i j

```

There is one binary operator defined on tuple-value expressions: broadcast (`^`).
This operator implements [numpy style broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

```
>>> check_shapes(
...     a=(randn(1, 3), "*x"),
...     b=(randn(2, 1), "*y"),
...     c=(randn(2, 3, 4), "*(x ^ y) z"),
... )
{'x': (1, 3), 'y': (2, 1), 'z': 4}
>>> check_shapes(
...     a=(randn(3), "*x"),
...     b=(randn(2, 1), "*y"),
...     c=(randn(2, 4, 4), "*(x ^ y) z"),
... )
Traceback (most recent call last):
  ...
ValueError: c dims (0, 1): expected (x^y)=(2, 3) got (2, 4)
    x=(3,)
    y=(2, 1)
    z=4
  a: got (3,)      expected [*x]
  b: got (2, 1)    expected [*y]
  c: got (2, 4, 4) expected [*(x^y) z]

```

### Determining Rank

In order for eincheck to test whether a Tensor's shape matches a shape spec it needs to be able to assign all parts of the Tensor's shape to the dimension specs.
Doing so requires there to be at most one repeated or unbound variadic  dimension.
For example, eincheck cannot compare `"i* j*"` or `"*i j*"` to a Tensor shape.
However, there can be multiple variadic dimensions if at least all but one is known.


```
>>> check_shapes((randn(3, 3, 4), "i* j*"))
Traceback (most recent call last):
  ...
ValueError: Unable to determine bindings for: arg0
  arg0: got (3, 3, 4) expected [i* j*]
>>> check_shapes((randn(3, 3, 4), "*i j*"))
Traceback (most recent call last):
  ...
ValueError: Unable to determine bindings for: arg0
  arg0: got (3, 3, 4) expected [*i j*]
>>> check_shapes((randn(3, 3, 4), "*i j*"), i=(3, 3))
{'i': (3, 3), 'j': 4}
>>> check_shapes(
...     z=(randn(3, 2, 4), "*(a ^ b) *c"),
...     a=(randn(1, 2), "*a"),
...     b=(randn(3, 1), "*b"),
...     c=(randn(4), "*c"),
... )
{'a': (1, 2), 'b': (3, 1), 'c': (4,)}

```

### Function Checks

The `check_func` decorator adds shape checks to the inputs and outputs of the decorated function.
The simplest way to specify shapes is as a comma separated list of input shape specs, an arrow `->`, and a comma separated list of output shape specs.

```
>>> @check_func("i, i -> i")
... def foo(x, y):
...     return x + y
>>> foo(randn(4), randn(4)).shape
(4,)
>>> foo(randn(4), randn(3))
Traceback (most recent call last):
  ...
ValueError: y dim 0: expected i=4 got 3
    i=4
  x: got (4,) expected [i]
  y: got (3,) expected [i]

```
Input specs match function parameters in the order they're declared.
There need to be at least as many function parameters as input shape specs.

```
>>> @check_func("i -> i")
... def foo(x, y):
...     return x + y

>>> foo(randn(4), randn(4)).shape
(4,)
>>> foo(randn(3, 4), randn(1))
Traceback (most recent call last):
  ...
ValueError: x: expected rank 1, got shape (3, 4)
  x: got (3, 4) expected [i]
>>> foo(randn(3), randn(3, 3))
Traceback (most recent call last):
  ...
ValueError: output0: expected rank 1, got shape (3, 3)
  i = 3
  output0: got (3, 3) expected [i]
>>> @check_func("i, i, i -> i")
... def foo(x):
...     return x
Traceback (most recent call last):
  ...
ValueError: Expected at least 3 input parameters, got 1


```

The shape spec for a variadic positional argument (e.g. `*args`) or variadic keyword argument (e.g. `**kwargs`) is compared against each Tensor matching that argument.

```
>>> @check_func("*x -> _ *x")
... def stack(*x):
...     return np.stack(x, 0)
>>> stack(randn(3, 4), randn(3, 4)).shape
(2, 3, 4)
>>> stack(randn(3, 4), randn(3, 5), randn(3, 4))
Traceback (most recent call last):
  ...
ValueError: x_1 dims (0, 1): expected x=(3, 4) got (3, 5)
    x=(3, 4)
  x_0: got (3, 4) expected [*x]
  x_1: got (3, 5) expected [*x]
  x_2: got (3, 4) expected [*x]


```
Multiple output specs can be used if the function returns a tuple.


```

```

### Data Checks

## Shape Specification

The notation for shapes is inspired by other einstein notation functions.

### Dimensions and Variables

The shape of a Tensor is a space separated list of dims.
Integers and variables (made up of alphabet characters) can be used to specify each dimension.
Variables can be any string of `[a-zA-Z]+`.
The shape spec `"i 4 inner"` will match any 3D Tensor where the second dimension is 4.
Upon matching a given `shape`, the variable `i` will be assigned to `shape[0]` and the variable `inner` will be assigned to `shape[2]`.

Variables have a single meaning across all the specs being considered.
The shape spec `"i i"` matches any 2D Tensor with the same size of both dimensions (i.e. a square matrix).
Similarly, `check_shapes((x, "i j"), (y, "i k"))` will check that `x` and `y` are both 2D Tensors, and that `x.shape[0] == y.shape[0]`.


### Repeated and Variadic Dimensions
If the dimension is followed by a `*`, it is a repeated dimension.
The repeated dimension spec `d*` matches any number of dimensions that all have size `d`.
For example, the spec `"batch 3*"` will check that `len(shape) >= 1 and all(x == 3 for x in shape[1:])`.

If the dimension is preceded by a `*`, it is a variadic dimension.
The variadic dimension spec `*d` matches any number of dimensions with the variable `d` a tuple of integers.
For example, `"*n i i"` will check that `len(shape) >= 2 and shape[-1] == shape[-2]`.
Furthermore, `i` will evaluate to `shape[-1]` and `n` will evaluate to `shape[:-2]`.

### Underscore and Ellipse
The special character `_` denotes a dimension with no constraints.
For example, `check_shapes((x, "i _"), (y, "i _"))` will check that `x` and `y` are both 2D and have the same first dimension, without making any assertion about the value of the second dimensions (`x.shape[1]` does not need to equal `y.shape[1]`).

The underscore can be combined with repeated or variadic specifications to denote an arbitrary of unconstrained dimensions.
The ellipsis is another option with the same meaning; `...`, `*_`, and `_*` all have the same meaning.
This can occur at any place in the shape spec. For example, `"8 ... 2 3"` checks that `len(shape) >= 3 and shape[0] == 8 and shape[-2:] == (2, 3)`.

### Expressions

In addition to a literal integer or a variable, a dimension can also be specified with an expression made up of integers, variables, and binary operators.
Expressions are always enclosed in parentheses.

The available ops are listed below.
Each op works with either integers or tuples.
Mixing data types (e.g. `(3 ^ a)`) will cause an exception.
In the examples below, `i = 7`, `a = (1, 2)`, and `b = (3, 4, 1)`.

| Op Name | Op Symbol | Data Type | Example |
| --- | --- | --- | --- |
| Add | `+` | Integers | `(i + 1) = 8` |
| Subtract | `-` | Integers | `(i - 4) = 3` |
| Multiply | `*` | Integers | `(2 * i) = 14` |
| Concat | `\|\|` | Tuples | `(a \|\| b) = (1, 2, 3, 4, 1)` |
| Broadcast | `^` | Tuples | `(a ^ b) = (3, 4, 2)` |

Expressions can be nested (e.g. `((2 * i) + 1) = (14 + 1) = 15`).

Expressions allow more complicated shape relations to be expressed. For example,
```
x: npt.NDArray
y: npt.NDArray

check_shapes(
... (x, "*batch x"),
... (y, "*batch y"),
... (np.concatenate([x, y], axis=-1), "*batch (x + y)"),
)

check_shapes((x, "*x"), (y, "*y"), (x + y, "*(x ^ y)"))

```
### Other types of ShapeArg

TODO

## The Tensor Type

## Check Shapes

The `check_shapes` function takes pairs of `(Tensor, shape)` to check.

These pairs can either be passed as positional or keyword arguments.
Positional arguments will be named `arg0`, `arg1`, etc.; keyword arguments will use the keyword name.

Additional keyword arguments can specify the values of variables.
The function returns the variable values found while checking shapes.
These can be used to chain shape checks.

```
shape_vars = check_shapes(
    x=(x, "*batch i j"), y=(y, "*batch j k"),
)
z = np.matmul(x, y)
check_shapes((z, "*batch i k"), **shape_vars)

```
## Check Functions

The `check_func` decorator adds shape checks before and after the decorated function.

Shapes can be specified with positional order or keywords.

If the `shapes` argument contains `->`, it will be split into `{input_shapes} -> {output_shapes}`.
If no arrow is present, `shapes` will be treated as only output shapes (as if `-> ` is prepended).

The `input_shapes` and `output_shapes` are comma separated lists of shape specs.
The input shape specs will be matched with the function arguments in the order from the function signature.
If the function has a single output, there should be only one output shape.
If the function has multiple outputs (i.e. a tuple of Tensors), the output shape specs will be matched against the function outputs.

Input shapes can also be specified as keyword arguments to `check_func`.
A given function parameter should not have both a positional and keyword shape spec.

Example: positional and keyword shape specs.

```
@check_func("i j, j k -> i k")
def matmul(x, y):
  return x @ y

# Equivalent to matmul using keyword params.
@check_func("i k", y="j k", x="i j")
def matmul2(x, y):
  return x @ y

# Equivalent to matmul using both positional and keyword params.
@check_func("i j -> i k", y="j k")
def matmul3(x, y):
  return x @ y

```
Example: multiple outputs
```
@check_func("a, b -> a, a b, a b a")
def foo(a, b):
  """
  This shape check will check the shapes of x0, x1, and x2 but not x3.
  """
  x0 = a
  x1 = np.expand_dims(x0, -1) * b
  x2 = np.expand_dims(x1, -1) * a
  x3 = np.expand_dims(x2, -1) * b
  return x0, x1, x2, x3

```

## Check Data