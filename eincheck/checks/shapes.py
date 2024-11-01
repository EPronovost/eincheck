import itertools
from typing import Any, Dict, Optional, Set, Tuple, Union, cast

from eincheck.contexts import _should_do_checks
from eincheck.parser.dim_spec import DimSpec, DimType
from eincheck.parser.expressions import DataExpr, Variable
from eincheck.parser.grammar import ShapeArg, create_shape_spec
from eincheck.parser.shape_spec import ShapeSpec
from eincheck.types import ShapeVariable, Tensor
from eincheck.utils import get_shape


def _check_dim_spec(
    got_shape: Tuple[int, ...],
    d: DimSpec,
    bindings: Dict[str, ShapeVariable],
    name: str,
    msg: str,
    start_idx: int,
) -> None:
    if d.value is None:
        return

    expected_value = d.value.eval(bindings)

    if d.can_broadcast:
        broadcast_values: Set[ShapeVariable]
        if isinstance(expected_value, int):
            broadcast_values = {expected_value, 1}
        else:
            broadcast_values = {
                tuple(p) for p in itertools.product(*([x, 1] for x in expected_value))
            }

    if d.type is DimType.VARIADIC and isinstance(expected_value, int):
        raise ValueError(
            f"{name}: expected variadic DimSpec {d} to evaluate to a tuple, "
            f"got {expected_value}{msg}"
        )
    elif d.type is not DimType.VARIADIC and isinstance(expected_value, tuple):
        raise ValueError(
            f"{name}: expected non-variadic DimSpec {d} to evaluate to an integer, "
            f"got {expected_value}{msg}"
        )

    def do_check(g: ShapeVariable, indices: ShapeVariable) -> None:
        if d.can_broadcast and g not in broadcast_values:
            first_line = f"expected can broadcast to {d.value}={expected_value} got {g}"
        elif not d.can_broadcast and g != expected_value:
            first_line = f"expected {d.value}={expected_value} got {g}"
        else:
            first_line = None

        if first_line:
            dim_str = (
                f"dim {indices}" if isinstance(indices, int) else f"dims {indices}"
            )
            raise ValueError(
                f"{name} {dim_str}: {first_line}"
                + "".join(f"\n    {k}={v}" for k, v in bindings.items())
                + msg
            )

    if d.type is DimType.REPEATED:
        for g_idx, g in enumerate(got_shape):
            do_check(g, start_idx + g_idx)
    elif d.type is DimType.VARIADIC:
        do_check(got_shape, tuple(range(start_idx, start_idx + len(got_shape))))
    elif len(got_shape) != 1:
        raise RuntimeError(
            f"{name}: expected a single dimension for {d}, got {got_shape}{msg}"
        )
    else:
        do_check(got_shape[0], start_idx)


def _bind_shape(
    got_shape: Tuple[Optional[int], ...],
    s: ShapeArg,
    bindings: Dict[str, ShapeVariable],
    name: str,
    msg: str,
) -> None:
    expected_shape = create_shape_spec(s)

    for d, start_idx, end_idx in expected_shape.matched_indices(
        bindings, len(got_shape)
    ):
        if not isinstance(d.value, Variable) or d.value.x in bindings:
            continue

        g_slice = got_shape[start_idx:end_idx]
        if None in g_slice:
            raise ValueError(
                f"{name}: tried to match {g_slice} to {d}, found None{msg}"
            )

        g_slice = cast(Tuple[int, ...], g_slice)

        if d.can_broadcast:
            pass
        elif d.type is DimType.VARIADIC:
            bindings.setdefault(d.value.x, g_slice)
        elif d.type is DimType.REPEATED and g_slice:
            bindings.setdefault(d.value.x, g_slice[0])
        elif d.type is DimType.SINGLE and len(g_slice) != 1:
            raise RuntimeError(
                f"{name}: expected a single dimension for {d}, got {got_shape}{msg}"
            )
        elif d.type is DimType.SINGLE:
            bindings.setdefault(d.value.x, g_slice[0])
        else:
            # Only reach here if d.type is REPEATED and not got_shape.
            # Nothing to check in this case.
            pass


def _check_shape(
    got_shape: Tuple[Optional[int], ...],
    s: ShapeArg,
    bindings: Dict[str, ShapeVariable],
    name: str,
    msg: str,
) -> Dict[str, ShapeVariable]:
    expected_shape = create_shape_spec(s)

    if msg:
        msg = "\n" + msg

    _check_rank(name, got_shape, expected_shape, bindings, msg)

    unknown_size_inds = expected_shape.unknown_n_dims_indices(bindings)
    if len(unknown_size_inds) > 1:
        raise RuntimeError(
            f"{name} has multiple DimSpec of unknown size: "
            f"{[expected_shape.dims[i] for i in unknown_size_inds]}" + msg
        )

    for d, start_idx, end_idx in expected_shape.matched_indices(
        bindings, len(got_shape)
    ):
        if d.value is None:
            continue

        if d.type is DimType.REPEATED and start_idx == end_idx:
            continue

        g_slice = got_shape[start_idx:end_idx]
        if None in g_slice:
            inds = (
                f"dim {start_idx}"
                if end_idx == start_idx + 1
                else f"dims {tuple(range(start_idx, end_idx))}"
            )
            raise ValueError(
                f"{name} {inds}: tried to check {d} against {g_slice}, found None{msg}"
            )

        _check_dim_spec(
            cast(Tuple[int, ...], g_slice), d, bindings, name, msg, start_idx
        )

    return bindings


def check_shapes(
    *args: Tuple[Tensor, ShapeArg],
    **kwargs: Union[ShapeVariable, Tuple[Tensor, ShapeArg]],
) -> Dict[str, ShapeVariable]:
    """Check the shapes of Tensors against ShapeArg specifications.

    Examples:

    .. doctest::

        >>> from numpy.random import randn
        >>> from eincheck import check_shapes
        >>>
        >>> check_shapes((randn(3, 4, 5), "... i j"), (randn(5, 6), "... j k"))
        {'i': 4, 'j': 5, 'k': 6}
        >>> check_shapes(
        ...     x=(randn(8, 2, 7, 3), "*batch t 3"),
        ...     y=(randn(8, 2, 1, 1, 3), "*batch ... 3"),
        ...     batch=(8, 2),
        ... )
        {'batch': (8, 2), 't': 7}

    Pass pairs of (tensor, shape spec) as either args or kwargs. The only difference
    when using a kwarg is the display name in error messages (args are called ``arg0``,
    ``arg1``, etc).

    Kwargs can also specify the variable values (e.g. ``batch=(8, 2)``).

    If all shape checks pass, returns a dictionary with all variable values.

    :param args: Pairs of (tensor, shape spec)
    :param kwargs: Pairs of (tensor, shape spec)
    :raise ValueError: If the shapes are incorrect or cannot be verified
    :return: Values for all bound variables from the shape specs
    """
    if not _should_do_checks():
        return {}

    tensors, bindings = _get_tensors_and_bindings(*args, **kwargs)
    _check_variable_types(tensors, bindings)

    if not tensors:
        return bindings

    got_msgs = [str(x) for x, _ in tensors.values()]
    got_len = max(map(len, got_msgs))
    msg = "\n".join(
        f"  {name}: got {g:<{got_len}} expected {s}"
        for (name, (_, s)), g in zip(tensors.items(), got_msgs)
    )

    checked_names = set()
    binded_names = set()

    for _ in range(len(tensors)):
        bindings_len = len(bindings)

        for t_name, (t_got, t_expected) in tensors.items():
            if (
                t_name in checked_names
                or len(t_expected.unknown_n_dims_indices(bindings)) > 1
            ):
                continue

            if t_name not in binded_names:
                _check_rank(t_name, t_got, t_expected, bindings, msg)
                _bind_shape(t_got, t_expected, bindings, t_name, msg)
                binded_names.add(t_name)

            if not t_expected.is_checkable(bindings):
                continue

            _check_shape(t_got, t_expected, bindings, t_name, msg)
            checked_names.add(t_name)

        if len(bindings) == bindings_len:
            break

    unbound = set(tensors) - binded_names
    if unbound:
        raise ValueError(
            f"Unable to determine bindings for: {' '.join(unbound)}\n{msg}"
        )
    unchecked = set(tensors) - checked_names
    if unchecked:
        missing_vars = set.union(*(tensors[k][1].variables for k in unchecked))
        raise ValueError(
            f"Unable to check: [{' '.join(sorted(unchecked))}] "
            f"missing variables: [{' '.join(sorted(missing_vars))}]\n{msg}"
        )

    return bindings


def _check_variable_types(
    tensors: Dict[str, Tuple[Tuple[Optional[int], ...], ShapeSpec]],
    bindings: Dict[str, ShapeVariable],
) -> None:
    """Check that each variable is either an int or a tuple.

    Categorizes each variable present in the ShapeSpec and bindings as being either an
    int or a tuple.

    If any variables are in both sets, raises an error.
    """

    int_vars = set()
    tuple_vars = set()

    for k, v in bindings.items():
        if isinstance(v, int):
            int_vars.add(k)
        else:
            tuple_vars.add(k)

    for _, spec in tensors.values():
        for dim in spec.dims:
            if not dim.value:
                continue

            if dim.type is DimType.VARIADIC:
                tuple_vars.update(dim.value.variables)
            else:
                int_vars.update(dim.value.variables)

    both = int_vars & tuple_vars
    if both:
        raise ValueError(
            "Found variables in both variadic and non-variadic expressions: "
            + " ".join(sorted(both))
        )


def _check_rank(
    name: str,
    got_shape: Tuple[Optional[int], ...],
    shape_spec: ShapeSpec,
    bindings: Dict[str, ShapeVariable],
    msg: str,
) -> None:
    expected_rank = shape_spec.min_rank(bindings)

    if shape_spec.unknown_n_dims_indices(bindings):
        bound_text = "at least "
        check = len(got_shape) < expected_rank
    else:
        bound_text = ""
        check = len(got_shape) != expected_rank

    if check:
        rows = [
            f"{name}: expected rank {bound_text}{expected_rank}, "
            f"got shape {got_shape}"
        ] + [f"  {k} = {v}" for k, v in bindings.items()]
        raise ValueError("\n".join(rows) + "\n" + msg)


def _get_tensors_and_bindings(
    *args: Tuple[Tensor, ShapeArg],
    **kwargs: Union[ShapeVariable, Tuple[Tensor, ShapeArg]],
) -> Tuple[
    Dict[str, Tuple[Tuple[Optional[int], ...], ShapeSpec]], Dict[str, ShapeVariable]
]:
    tensors: Dict[str, Tuple[Tuple[Optional[int], ...], ShapeSpec]] = {}

    for idx, (a_tensor, a_shape) in enumerate(args):
        tensors.update(_get_shapes(a_tensor, create_shape_spec(a_shape), f"arg{idx}"))

    bindings: Dict[str, ShapeVariable] = {}

    for k, v in kwargs.items():
        if isinstance(v, int) or (
            isinstance(v, tuple) and all(isinstance(vv, int) for vv in v)
        ):
            bindings[k] = cast(ShapeVariable, v)
        elif isinstance(v, tuple) and len(v) == 2:
            v = cast(Tuple[Tensor, ShapeArg], v)
            assert k not in tensors
            tensors.update(_get_shapes(v[0], create_shape_spec(v[1]), k))
        else:
            raise ValueError(f"Unexpected kwarg {v}")

    return tensors, bindings


def _get_shapes(
    x: Any, s: ShapeSpec, name: str
) -> Dict[str, Tuple[Tuple[Optional[int], ...], ShapeSpec]]:
    if s.is_data_expr:
        if not hasattr(x, "_get_shapes"):
            raise ValueError(
                f"{name}: spec $ specified, but no _get_shapes method was found. "
                "This should have been added by the @check_data decorator."
            )

        return {
            k2: v2
            for k, (vt, vs) in x._get_shapes().items()
            for k2, v2 in _get_shapes(vt, vs, f"{name}.{k}").items()
        }

    shape = get_shape(x)
    if shape is not None:
        if any(isinstance(d.value, DataExpr) for d in s.dims):
            raise ValueError(
                f"{name}: $ should not be present in the shape spec for a Tensor, "
                f"got {s}"
            )
        return {name: (shape, s)}

    return {}
