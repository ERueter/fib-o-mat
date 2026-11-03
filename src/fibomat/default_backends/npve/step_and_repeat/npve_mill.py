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

import typing as t
import warnings

from fibomat.mill.dose_mill import Dose, DoseMill
from fibomat.units import DimFloat, TimeDimension, check_units


__all__ = ['NPVEMill', 'Dose']


_SCAN_DIRECTION_DEPRECATION = (
    'scan_direction of NPVEMill is deprecated and will be removed in a future version.'
)


class NPVEMill(DoseMill):
    """Mill of the NPVE step and repeat backend.

    The dwell time is required. Either the number of repeats or the dose is given (not both): with `repeats`, the
    pattern is exposed `repeats` times, with a `dose` NPVE calculates the number of repeats. The dose is a charge
    (``unit('ions')``) for spots, a charge per length for curves (e.g. ``unit('nC/µm')``) and a charge per area for
    areas (e.g. ``unit('nC/µm**2')``), see :class:`~fibomat.mill.DoseMill`.
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
            scan_direction (float, optional): deprecated. Rotation of the scan direction. 0: right to left, pi/2:
                bottom to top.
            **kwargs: further settings of the mill

        Raises:
            DimensionError: Raised if dwell_time or dose are no dimensioned values or have a wrong dimension.
            TypeError: Raised if repeats is no integer.
            ValueError: Raised if repeats and dose are both given or both missing, dwell_time is not a positive time,
                repeats is less than 1 or dose is not a positive dose.
        """
        if (repeats is not None and dose is not None) or (repeats is None and dose is None):
            raise ValueError('One of repeats or dose must be given and not both or none of them.')

        if scan_direction is not None:
            warnings.warn(_SCAN_DIRECTION_DEPRECATION, DeprecationWarning, stacklevel=3)
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
    def scan_direction(self) -> t.Optional[float]:
        """Rotation of the scan direction (0: right to left, pi/2: bottom to top), None if it is not set.

        Deprecated: ``scan_direction`` will be removed in a future version.

        Access:
            get
        """
        warnings.warn(_SCAN_DIRECTION_DEPRECATION, DeprecationWarning, stacklevel=2)
        return self._scan_direction


def scan_direction_of(mill: NPVEMill) -> t.Optional[float]:
    """The scan direction of a mill, without a deprecation warning (used by the step and repeat backend).

    Args:
        mill (NPVEMill): mill

    Returns:
        Optional[float]: scan direction or None
    """
    return mill._scan_direction  # pylint: disable=protected-access
