"""Provides the :class:`NPVEMill` class.

Example:
    >>> from fibomat.units import unit
    >>> mill = NPVEMill(dwell_time=1. * unit('µs'), repeats=3)
    >>> mill.repeats, mill.dose
    (3, None)
    >>> mill = NPVEMill(dwell_time=1. * unit('µs'), dose=2. * unit('nC / µm**2'))
    >>> mill.repeats is None
    True
"""
from __future__ import annotations

import operator
import typing as t

from fibomat.mill import MillBase
from fibomat.units import (
    AreaDoseDimension, DimFloat, LineDoseDimension, SpotDoseDimension, TimeDimension, check_units
)


__all__ = ['NPVEMill']


Dose = t.Union[DimFloat[SpotDoseDimension], DimFloat[LineDoseDimension], DimFloat[AreaDoseDimension]]
"""A dose: a charge (spot), a charge per length (curve) or a charge per area (area)."""


class NPVEMill(MillBase):
    """Mill of the NPVE step and repeat backend.

    Either the number of repeats or the dose is given (not both): with `repeats`, the pattern is exposed `repeats`
    times, with a `dose` NPVE calculates the number of repeats. The dose is a charge (``unit('ions')``) for spots, a
    charge per length for curves (e.g. ``unit('nC/µm')``) and a charge per area for areas (e.g. ``unit('nC/µm**2')``).
    """

    @check_units
    def __init__(
        self,
        *,
        dwell_time: DimFloat[TimeDimension],
        repeats: t.Optional[int] = None,
        dose: t.Optional[Dose] = None,
        scan_direction: t.Optional[float] = None,
        **kwargs: t.Any,
    ):
        """
        Args:
            dwell_time (DimFloat): dwell time per spot, e.g. ``1. * unit('µs')``
            repeats (int, optional): number of repeats (at least 1)
            dose (DimFloat, optional): dose (see above)
            scan_direction (float, optional): rotation of the scan direction. 0: right to left, pi/2: bottom to top.
            **kwargs: further settings of the mill

        Raises:
            DimensionError: Raised if dwell_time or dose are no dimensioned values or have a wrong dimension.
            TypeError: Raised if repeats is no integer.
            ValueError: Raised if repeats and dose are both given or both missing, dwell_time is not a positive time,
                repeats is less than 1 or dose is not a positive dose.
        """
        if (repeats is not None and dose is not None) or (repeats is None and dose is None):
            raise ValueError('One of repeats or dose must be given and not both or none of them.')

        if not dwell_time.magnitude > 0.:
            raise ValueError('dwell_time must be positive.')

        if repeats is not None:
            if isinstance(repeats, bool):
                raise TypeError('repeats must be an int.')
            try:
                repeats = operator.index(repeats)
            except TypeError:
                raise TypeError(f'repeats must be an int, got {type(repeats).__name__}.') from None
            if repeats < 1:
                raise ValueError('repeats must be at least 1.')

        if dose is not None and not dose.magnitude > 0.:
            raise ValueError('dose must be positive.')

        if scan_direction is not None:
            scan_direction = float(scan_direction)

        self._scan_direction = scan_direction

        super().__init__(dwell_time=dwell_time, repeats=repeats, dose=dose, **kwargs)

    @property
    def dwell_time(self) -> DimFloat[TimeDimension]:
        """Dwell time per spot.

        Access:
            get
        """
        return self['dwell_time']

    @property
    def repeats(self) -> t.Optional[int]:
        """Number of repeats, None if the dose is given.

        Access:
            get
        """
        return self._kwargs['repeats']

    @property
    def dose(self) -> t.Optional[Dose]:
        """Dose, None if the number of repeats is given.

        Access:
            get
        """
        return self._kwargs['dose']

    @property
    def scan_direction(self) -> t.Optional[float]:
        """Rotation of the scan direction (0: right to left, pi/2: bottom to top), None if it is not set.

        Access:
            get
        """
        return self._scan_direction
