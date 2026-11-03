"""Provides the :class:`ColorBy` option of the :class:`~fibomat.default_backends.svg_backend.SVGBackend`."""
from __future__ import annotations

import enum
import typing as t


__all__ = ['ColorBy']


E = t.TypeVar('E', bound=enum.Enum)


class ColorBy(str, enum.Enum):
    """Mill setting which is encoded by the color of the shapes.

    The backend accepts the members or their string values (``'dwell_time'`` or ``ColorBy.DWELL_TIME``).
    """

    DWELL_TIME = 'dwell_time'
    """Dwell time per spot (``mill['dwell_time']``)."""
    REPEATS = 'repeats'
    """Number of repeats (``mill['repeats']``)."""
    DOSE = 'dose'
    """Dose (``mill['dose']``, e.g. of a :class:`~fibomat.mill.DoseMill`)."""

    @classmethod
    def parse(cls: t.Type[E], value: t.Union[E, str]) -> E:
        """Return the member for `value` (a member or its string value).

        Args:
            value (ColorBy, str): member or value

        Returns:
            ColorBy

        Raises:
            ValueError: Raised if `value` is unknown.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            options = ', '.join(repr(member.value) for member in cls)  # type: ignore[attr-defined]
            raise ValueError(f'Unknown {cls.__name__} {value!r}, expected one of {options}.') from None
