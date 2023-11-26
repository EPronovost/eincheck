import functools
import inspect
from typing import Any, Callable, TypeVar

from eincheck.checks.shapes import check_shapes
from eincheck.contexts import _should_do_checks
from eincheck.parser.grammar import ShapeArg, create_shape_spec

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

        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if not _should_do_checks():
                return func(*args, **kwargs)

            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            input_data = {}
            for p in sig.parameters.values():
                if p.name not in input_shapes:
                    continue

                x = bound_args.arguments[p.name]
                if p.kind is inspect.Parameter.VAR_POSITIONAL:
                    assert isinstance(x, tuple)
                    p_data = [(f"{p.name}_{x_idx}", xx) for x_idx, xx in enumerate(x)]
                elif p.kind is inspect.Parameter.VAR_KEYWORD:
                    assert isinstance(x, dict)
                    p_data = list(x.items())
                else:
                    p_data = [(p.name, x)]

                for x_name, xx in p_data:
                    assert x_name not in input_data
                    input_data[x_name] = (xx, input_shapes[p.name])

            updated_spec = check_shapes(**input_data)

            out = func(*args, **kwargs)

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
            return out

        return inner  # type: ignore[return-value]

    return wrapper
