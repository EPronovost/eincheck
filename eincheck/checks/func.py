import functools
import inspect
from typing import Any, Callable, Dict, Iterable, List, Mapping, TypeVar, Union

from eincheck.checks.shapes import check_shapes
from eincheck.contexts import _should_do_checks
from eincheck.parser.grammar import ShapeArg, create_shape_spec
from eincheck.parser.shape_spec import ShapeSpec
from eincheck.utils import get_object, parse_dot_name

_T_Callable = TypeVar("_T_Callable", bound=Callable[..., Any])


def check_func(
    shapes: str = "", **kwargs: ShapeArg
) -> Callable[[_T_Callable], _T_Callable]:
    """Check the input and output shapes of a function.


    :param shapes: string of input and output shape specs
    :param kwargs: additional shape specs for function inputs
    :return: a function decorator
    """
    if "->" in shapes:
        input_str, output_str = shapes.split("->", 2)
    else:
        input_str = ""
        output_str = shapes

    input_arg_shapes = [
        create_shape_spec(s.strip()) for s in input_str.split(",") if s.strip()
    ]
    input_kwarg_shapes = {k: create_shape_spec(v) for k, v in kwargs.items()}

    output_shapes = [
        create_shape_spec(s.strip()) for s in output_str.split(",") if s.strip()
    ]

    return _get_wrapper(input_arg_shapes, input_kwarg_shapes, output_shapes)


def check_func2(
    input_shapes: Union[str, Mapping[str, ShapeArg]],
    output_shapes: Union[str, Mapping[str, ShapeArg]] = "",
) -> Callable[[_T_Callable], _T_Callable]:
    """Check the input and output shapes of a function.

    This function is an alternative to ``check_func`` that works better with
    dictionaries.
    It takes an input spec and an output spec, where each spec can be either a
    dictionary or a comma separated string.

    If both are strings, ``check_func2(input_str, output_str)`` is equivalent to
    ``check_func(f"{input_str} -> {output_str}")``.

    If the input spec is a dictionary and the output spec is a string,
    ``check_func2(input_dict, output_str)`` is equivalent to
    ``check_func(output_str, **input_dict)``.

    This decorator also supports a dictionary for output shapes, which ``check_func``
    does not.
    This enables dotpath names on the returned object.

    Examples:

    .. doctest::

        >>> from eincheck import check_func2
        >>> from numpy.random import randn
        >>> from numpy.typing import NDArray
        >>> from typing import NamedTuple, Tuple
        >>>
        >>> Array = NDArray[float]
        >>>
        >>> # Three equivalent ways of using check_func2.
        >>> @check_func2("i, j", "i j, i j")
        ... def foo1(x: Array, y: Array) -> Tuple[Array, Array]:
        ...     return x[:, None] + y, x[:, None] * y
        ...
        >>> _ = foo1(randn(4), randn(5))
        >>>
        >>> @check_func2("i, j -> i j, i j")
        ... def foo2(x: Array, y: Array) -> Tuple[Array, Array]:
        ...     return x[:, None] + y, x[:, None] * y
        ...
        >>> _ = foo2(randn(4), randn(5))
        >>>
        >>> @check_func2({"x": "i", "y": "j"}, {"0": "i j", "1": "i j"})
        ... def foo3(x: Array, y: Array) -> Tuple[Array, Array]:
        ...     return x[:, None] + y, x[:, None] * y
        ...
        >>> _ = foo3(randn(4), randn(5))
        >>>
        >>> class Pair(NamedTuple):
        ...     first: Array
        ...     second: Array
        ...
        >>> @check_func2(
        ...     {"x.first": "*a", "x.second": "*b", "y.first": "*a", "y.second": "*b"},
        ...     {"first": "*a", "second": "*b"},
        ... )
        ... def add_pairs(x: Pair, y: Pair) -> Pair:
        ...     return Pair(x.first + y.first, x.second + y.second)
        ...
        >>> _ = add_pairs(Pair(randn(4), randn(5, 6)), Pair(randn(4), randn(5, 6)))

    :param input_shapes: comma separated string or dictionary of shapes
    :param output_shapes: comma separated string or dictionary of shapes
    :return: a function decorator
    """
    if isinstance(input_shapes, str) and "->" in input_shapes:
        if output_shapes:
            raise ValueError(
                "'->' in input_shapes should only be used when output_shapes is empty"
            )

        input_shapes, output_shapes = input_shapes.split("->", 2)

    if isinstance(input_shapes, str):
        input_arg_shapes = [
            create_shape_spec(s.strip()) for s in input_shapes.split(",") if s.strip()
        ]
        input_kwarg_shapes = {}
    else:
        input_arg_shapes = []
        input_kwarg_shapes = {k: create_shape_spec(v) for k, v in input_shapes.items()}

    parsed_output_shapes: Union[List[ShapeSpec], Dict[str, ShapeSpec]]
    if isinstance(output_shapes, str):
        parsed_output_shapes = [
            create_shape_spec(s.strip()) for s in output_shapes.split(",") if s.strip()
        ]
    else:
        parsed_output_shapes = {
            k: create_shape_spec(v) for k, v in output_shapes.items()
        }

    return _get_wrapper(input_arg_shapes, input_kwarg_shapes, parsed_output_shapes)


def _get_wrapper(
    input_arg_shapes: List[ShapeSpec],
    input_kwarg_shapes: Dict[str, ShapeSpec],
    output_shapes: Union[List[ShapeSpec], Dict[str, ShapeSpec]],
) -> Callable[[_T_Callable], _T_Callable]:
    def wrapper(func: _T_Callable) -> _T_Callable:
        input_shapes = input_kwarg_shapes

        sig = inspect.signature(func)
        if len(sig.parameters) < len(input_arg_shapes):
            raise ValueError(
                f"Expected at least {len(input_arg_shapes)} input parameters, "
                f"got {len(sig.parameters)}"
            )

        for arg_spec, arg_name in zip(input_arg_shapes, sig.parameters):
            if arg_name in input_shapes:
                raise ValueError(
                    f"Spec for {arg_name} specified in both args and kwargs."
                )

            input_shapes[arg_name] = arg_spec

        # dot_name, name_base, name_parts
        # e.g. ("foo.x.y", "foo", ["x", "y"])
        parsed_names = [(n, *parse_dot_name(n)) for n in input_shapes]
        # sort to match signature for nice error messages
        sig_params = list(sig.parameters)
        parsed_names.sort(
            key=lambda t: (
                sig_params.index(t[1]) if t[1] in sig_params else len(sig_params)
            )
        )
        _check_no_extra_params((x for _, x, _ in parsed_names), sig)

        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if not _should_do_checks():
                return func(*args, **kwargs)

            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            input_data = {}
            for spec_name, spec_base, spec_parts in parsed_names:
                if spec_base not in sig.parameters:
                    continue

                p = sig.parameters[spec_base]
                x = get_object(spec_name, bound_args.arguments)

                if len(spec_parts) > 0:
                    p_data = [(spec_name, x)]
                elif p.kind is inspect.Parameter.VAR_POSITIONAL:
                    assert isinstance(x, tuple)
                    p_data = [
                        (f"{spec_name}_{x_idx}", xx) for x_idx, xx in enumerate(x)
                    ]
                elif p.kind is inspect.Parameter.VAR_KEYWORD:
                    assert isinstance(x, dict)
                    p_data = list(x.items())
                else:
                    p_data = [(spec_name, x)]

                for x_name, xx in p_data:
                    assert x_name not in input_data
                    input_data[x_name] = (xx, input_shapes[spec_name])

            updated_spec = check_shapes(**input_data)

            out = func(*args, **kwargs)

            if isinstance(output_shapes, list):
                out_tup = (
                    out if isinstance(out, tuple) and len(output_shapes) > 1 else (out,)
                )
                if len(out_tup) < len(output_shapes):
                    raise ValueError(
                        f"Expected at least {len(output_shapes)} outputs, "
                        f"got {len(out_tup)}"
                    )
                check_shapes(
                    **{
                        f"output{i}": (x, s)
                        for i, (x, s) in enumerate(zip(out_tup, output_shapes))
                    },
                    **updated_spec,
                )
            else:
                output_data = {
                    f"output {k}": (get_object(k, out), v)
                    for k, v in output_shapes.items()
                }
                check_shapes(**output_data, **updated_spec)

            return out

        return inner  # type: ignore[return-value]

    return wrapper


def _check_no_extra_params(got_names: Iterable[str], sig: inspect.Signature) -> None:
    """Check that all of got_names are valid for sig."""
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        # Any param name is valid.
        return

    extra_names = set(got_names) - set(sig.parameters)
    if extra_names:
        raise NameError(
            f"Parameter names not found in function signature: {extra_names}"
        )
