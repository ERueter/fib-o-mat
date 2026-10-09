"""Runtime validation of dimensions (:func:`check_units`)."""
from __future__ import annotations

import functools
import inspect
import types
import typing as t

from fibomat.units.dim_base import DimBase
from fibomat.units.dimensions import Dimension, DimensionError


__all__ = ['check_units']


_Checker = t.Callable[[str, t.Any], None]
"""Function ``check(label, value)`` which raises a :class:`DimensionError` for invalid values."""


def _dim_checker(origin: type, dimension: t.Type[Dimension]) -> _Checker:
    """Checker for a value annotated as ``SomeDimType[SomeDimension]``."""
    def check(label: str, value: t.Any) -> None:
        if not isinstance(value, origin):
            raise DimensionError(
                f'{label!r}: expected a {origin.__name__}[{dimension.__name__}], got {type(value).__name__}'
            )

        for suffix, quantity in value._dim_quantities():  # pylint: disable=protected-access
            if not dimension.matches(quantity):
                where = label if not suffix else f'{label}.{suffix}'
                raise DimensionError(
                    f'{where!r}: expected dimension {dimension.pint_dimension!r} ({dimension.__name__}), '
                    f'got {quantity.dimensionality} (units: {quantity.units})'
                )

    return check


def _build_checker(hint: t.Any) -> t.Optional[_Checker]:
    """Create a checker for an annotation or None if the annotation contains nothing which can be validated.

    Supported are ``SomeDimType[SomeDimension]`` (:class:`DimFloat` or any other :class:`DimBase` subclass),
    ``Optional[...]``, ``Union[...]`` of such annotations (the value must be valid for one of them) and ``Tuple[...]``
    (``Tuple[A, B]`` or ``Tuple[A, ...]``) of them. A union is only checked if all its members can be checked.
    """
    origin = t.get_origin(hint)
    args = t.get_args(hint)

    if origin is None or not args:
        return None

    if isinstance(origin, type) and issubclass(origin, DimBase):
        dimension = args[0]
        if isinstance(dimension, type) and issubclass(dimension, Dimension):
            return _dim_checker(origin, dimension)
        return None

    if origin is t.Union or origin is getattr(types, 'UnionType', t.Union):
        members = [arg for arg in args if arg is not type(None)]
        has_none = len(members) != len(args)
        checkers = [_build_checker(member) for member in members]
        if not checkers or any(checker is None for checker in checkers):
            return None

        def check_union(label: str, value: t.Any) -> None:
            if value is None and has_none:
                return
            errors = []
            for checker in checkers:
                try:
                    checker(label, value)  # type: ignore[misc]
                    return
                except DimensionError as error:
                    errors.append(str(error))
            raise DimensionError(' or '.join(errors))

        return check_union

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            item_checker = _build_checker(args[0])
            if item_checker is None:
                return None

            def check_variadic(label: str, value: t.Any) -> None:
                for i, item in enumerate(value):
                    item_checker(f'{label}[{i}]', item)

            return check_variadic

        item_checkers = [_build_checker(arg) for arg in args]
        if all(checker is None for checker in item_checkers):
            return None

        def check_tuple(label: str, value: t.Any) -> None:
            if len(value) != len(item_checkers):
                raise DimensionError(f'{label!r}: expected {len(item_checkers)} values, got {len(value)}')
            for i, (checker, item) in enumerate(zip(item_checkers, value)):
                if checker is not None:
                    checker(f'{label}[{i}]', item)

        return check_tuple

    return None


F = t.TypeVar('F', bound=t.Callable[..., t.Any])


def check_units(func: F) -> F:
    """Decorator which validates at call time the dimensions of all arguments and the return value.

    The annotations are evaluated on the first call (not when the function is decorated). Every parameter (and the
    return value) annotated as ``SomeDimType[SomeDimension]``, where ``SomeDimType`` is
    :class:`DimFloat` or any other :class:`DimBase` subclass, must hold a value whose embedded pint quantities have
    that dimensionality. This is independent of static type checking and catches values a type checker cannot verify
    (data loaded at runtime, `Any`, ``# type: ignore``, ...). ``Optional[...]``, ``Union[...]`` and ``Tuple[...]`` of
    such annotations are checked, too (``None`` is accepted for an ``Optional``).

    Args:
        func (Callable): function to be wrapped

    Returns:
        Callable: wrapped function

    Raises:
        DimensionError: (raised by the wrapper) a value does not have the annotated dimension.
    """
    sig = inspect.signature(func)
    # The annotations are resolved on the first call, because they may refer to names which are not defined yet when
    # the decorator is applied (e.g. a method of a class which returns this class).
    checks: t.List[t.Any] = []

    def resolve() -> None:
        hints = t.get_type_hints(func)
        param_checks = []
        for name, hint in hints.items():
            if name != 'return':
                checker = _build_checker(hint)
                if checker is not None:
                    param_checks.append((name, checker))
        checks.extend([param_checks, _build_checker(hints['return']) if 'return' in hints else None])

    @functools.wraps(func)
    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        if not checks:
            resolve()
        param_checks, return_check = checks

        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, checker in param_checks:
            if name in bound.arguments:
                checker(name, bound.arguments[name])

        result = func(*args, **kwargs)

        if return_check is not None:
            return_check('return value', result)

        return result

    return t.cast(F, wrapper)
