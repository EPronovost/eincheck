from typing import Dict, Optional, Tuple, Union, cast

from eincheck.parser.dim_spec import DimSpec, DimType
from eincheck.parser.expressions import Variable
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
        if g != expected_value:
            dim_str = (
                f"dim {indices}" if isinstance(indices, int) else f"dims {indices}"
            )
            raise ValueError(
                f"{name} {dim_str}: expected {d.value}={expected_value} got {g}"
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
                f"{name}: tried to assign {g_slice} to {d}, found None{msg}"
            )

        g_slice = cast(Tuple[int, ...], g_slice)

        if d.type is DimType.VARIADIC:
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

    _check_rank(name, got_shape, expected_shape, bindings, msg)

    if msg:
        msg = "\n" + msg

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
            raise ValueError(
                f"{name} tried to check shape {d} against {g_slice}, found None{msg}"
            )

        _check_dim_spec(
            cast(Tuple[int, ...], g_slice), d, bindings, name, msg, start_idx
        )

    return bindings


def check_shapes(
    *args: Tuple[Tensor, ShapeArg],
    **kwargs: Union[ShapeVariable, Tuple[Tensor, ShapeArg]],
) -> Dict[str, ShapeVariable]:
    tensors, bindings = _get_tensors_and_bindings(*args, **kwargs)
    if not tensors:
        return bindings

    got_msgs = [str(x) for x, _ in tensors.values()]
    got_len = max(map(len, got_msgs))
    msg = "\n".join(
        f"  {name}: got {g:<{got_len}} expected {s}"
        for (name, (_, s)), g in zip(tensors.items(), got_msgs)
    )

    checked_names = set()
    bindings_len = len(bindings)

    for _ in range(len(tensors)):
        for t_name, (t_got, t_expected) in tensors.items():
            if (
                t_name in checked_names
                or len(t_expected.unknown_n_dims_indices(bindings)) > 1
            ):
                continue

            _check_rank(t_name, t_got, t_expected, bindings, msg)
            _bind_shape(t_got, t_expected, bindings, t_name, msg)

            if not t_expected.is_checkable(bindings):
                continue

            _check_shape(t_got, t_expected, bindings, t_name, msg)
            checked_names.add(t_name)

        if len(bindings) == bindings_len:
            break

    unchecked = set(tensors) - checked_names
    if unchecked:
        raise ValueError(f"Unable to determine bindings for {unchecked}\n{msg}")

    return bindings


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
        raise ValueError("\n".join(rows) + msg)


def _get_tensors_and_bindings(
    *args: Tuple[Tensor, ShapeArg],
    **kwargs: Union[ShapeVariable, Tuple[Tensor, ShapeArg]],
) -> Tuple[
    Dict[str, Tuple[Tuple[Optional[int], ...], ShapeSpec]], Dict[str, ShapeVariable]
]:
    tensors: Dict[str, Tuple[Tuple[Optional[int], ...], ShapeSpec]] = {}

    for idx, (a_tensor, a_shape) in enumerate(args):
        s = get_shape(a_tensor)
        if s is not None:
            tensors[f"arg{idx}"] = (s, create_shape_spec(a_shape))

    bindings: Dict[str, ShapeVariable] = {}

    for k, v in kwargs.items():
        if isinstance(v, int) or (
            isinstance(v, tuple) and all(isinstance(vv, int) for vv in v)
        ):
            bindings[k] = cast(ShapeVariable, v)
        elif isinstance(v, tuple) and len(v) == 2:
            v = cast(Tuple[Tensor, ShapeArg], v)
            assert k not in tensors
            s = get_shape(v[0])
            if s is not None:
                tensors[k] = (s, create_shape_spec(v[1]))
        else:
            raise ValueError(f"Unexpected kwarg {v}")

    return tensors, bindings
