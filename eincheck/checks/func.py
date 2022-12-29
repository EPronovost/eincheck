import functools
import inspect
import itertools
from typing import Any, Callable, TypeVar

from eincheck.checks.shapes import check_shapes
from eincheck.parser.grammar import create_shape_spec
from eincheck.types import ShapeVariable

_T_Callable = TypeVar("_T_Callable", bound=Callable[..., Any])


def check_func(
    shapes: str, **spec: ShapeVariable
) -> Callable[[_T_Callable], _T_Callable]:
    input_str, output_str = shapes.split("->", 2)
    input_shapes = [
        create_shape_spec(s.strip()) for s in input_str.split(",") if s.strip()
    ]
    output_shapes = [
        create_shape_spec(s.strip()) for s in output_str.split(",") if s.strip()
    ]

    def wrapper(func: _T_Callable) -> _T_Callable:
        sig = inspect.signature(func)
        if len(sig.parameters) < len(input_shapes):
            raise ValueError(
                f"Expected at least {len(input_shapes)} input parameters, "
                f"got {len(sig.parameters)}"
            )

        is_method = inspect.ismethod(func)

        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            input_data = {}
            for p, s in zip(
                itertools.islice(sig.parameters.values(), is_method, None), input_shapes
            ):
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
                    input_data[x_name] = (xx, s)

            updated_spec = check_shapes(**input_data, **spec)

            out = func(*args, **kwargs)

            out_tup = (
                out if isinstance(out, tuple) and len(output_shapes) > 1 else (out,)
            )
            if len(out_tup) != len(output_shapes):
                raise ValueError(
                    f"Expected {len(output_shapes)} outputs, got {len(out_tup)}"
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
