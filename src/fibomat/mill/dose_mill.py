"""Provide the :class:`DoseMill` class and the :data:`Dose` type.

Example:
    >>> from fibomat.units import unit
    >>> mill = DoseMill(dwell_time=1. * unit('µs'), dose=2. * unit('ions / nm**2'))
    >>> mill.repeats is None
    True
    >>> mill['dose'].m_as('ions / nm**2')
    2.0
"""
from __future__ import annotations

import math
import operator
import typing as t

from fibomat.mill.mill_base import MillBase
from fibomat.units import AreaDoseDimension, DimFloat, LineDoseDimension, SpotDoseDimension, TimeDimension, check_units


__all__ = ['DoseMill', 'Dose']


Dose = t.Union[DimFloat[SpotDoseDimension], DimFloat[LineDoseDimension], DimFloat[AreaDoseDimension]]
"""A dose: a charge per spot (e.g. ``unit('ions')``), per length (curves, e.g. ``unit('nC / µm')``) or per area
(areas, e.g. ``unit('ions / nm**2')``)."""


def check_repeats(repeats: t.Any) -> int:
    """Check that `repeats` is an integer of at least 1.

    Args:
        repeats (Any): number of repeats

    Returns:
        int

    Raises:
        TypeError: Raised if `repeats` is no integer (booleans are rejected).
        ValueError: Raised if `repeats` is less than 1.
    """
    if isinstance(repeats, bool):
        raise TypeError('repeats must be an int.')
    try:
        repeats = operator.index(repeats)
    except TypeError:
        raise TypeError(f'repeats must be an int, got {type(repeats).__name__}.') from None
    if repeats < 1:
        raise ValueError('repeats must be at least 1.')
    return repeats


class DoseMill(MillBase):
    """Mill which is defined by two of the three settings dwell time, number of repeats and dose.

    Exactly two settings must be given; the third one is None. It is not calculated by the mill, because this needs
    the beam current and the pitches of the raster style. Backends which support the mill derive it (e.g. the NPVE
    software calculates the number of repeats from the dose).

    The settings are available as properties and as items (``mill['dose']``; the item of the missing setting raises a
    ``KeyError``).

    .. note:: The raster styles of fib-o-mat need a :class:`~fibomat.mill.Mill` (dwell time and repeats). A
        :class:`DoseMill` is meant for backends which pass the settings to the patterning software.
    """

    @check_units
    def __init__(
        self,
        *,
        dwell_time: t.Optional[DimFloat[TimeDimension]] = None,
        repeats: t.Optional[int] = None,
        dose: t.Optional[Dose] = None,
        **kwargs: t.Any,
    ):
        """
        Args:
            dwell_time (DimFloat[TimeDimension], optional): dwell time per spot, e.g. ``1. * unit('µs')``
            repeats (int, optional): number of repeats (at least 1)
            dose (Dose, optional): dose (see :data:`Dose`)
            **kwargs: further settings of the mill

        Raises:
            DimensionError: Raised if dwell_time or dose are no dimensioned values or have a wrong dimension.
            TypeError: Raised if repeats is no integer.
            ValueError: Raised if not exactly two of dwell_time, repeats and dose are given, dwell_time or dose is not
                positive and finite, or repeats is less than 1.
        """
        given = [name for name, value in (('dwell_time', dwell_time), ('repeats', repeats), ('dose', dose))
                 if value is not None]
        if len(given) != 2:
            raise ValueError(
                f'Exactly two of dwell_time, repeats and dose must be given, got {", ".join(given) or "none"}.'
            )

        for name, value in (('dwell_time', dwell_time), ('dose', dose)):
            if value is not None and not (math.isfinite(value.magnitude) and value.magnitude > 0.):
                raise ValueError(f'{name} must be positive and finite, got {value!r}.')

        if repeats is not None:
            repeats = check_repeats(repeats)

        super().__init__(dwell_time=dwell_time, repeats=repeats, dose=dose, **kwargs)

    @property
    def dwell_time(self) -> t.Optional[DimFloat[TimeDimension]]:
        """Dwell time per spot, None if it is not given.

        Access:
            get
        """
        return self._kwargs['dwell_time']

    @property
    def repeats(self) -> t.Optional[int]:
        """Number of repeats, None if it is not given.

        Access:
            get
        """
        return self._kwargs['repeats']

    @property
    def dose(self) -> t.Optional[Dose]:
        """Dose, None if it is not given.

        Access:
            get
        """
        return self._kwargs['dose']
