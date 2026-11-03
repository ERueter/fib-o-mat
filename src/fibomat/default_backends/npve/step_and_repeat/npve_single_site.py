"""Provides the :class:`NPVESingleSite` backend, which exports one site as NPVE shape file (``*.nps``).

Example::

    mill = NPVEMill(dwell_time=1. * unit('µs'), repeats=2)
    exported = layout.export(NPVESingleSite)
    exported.save('patterns.nps')
"""
from __future__ import annotations

import typing as t

import xmltodict

from fibomat.default_backends.npve.step_and_repeat.backend_base import (
    DEFAULT_MAX_EMBEDDED_POINTS, NPVEBackendBase
)
from fibomat.default_backends.npve.step_and_repeat.common_models import FIBShape
from fibomat.default_backends.npve.step_and_repeat.common_schemas import FIBShapeSchema
from fibomat.layout.site import Site
from fibomat.units import DimFloat, LengthDimension, check_units


__all__ = ['NPVESingleSite']


_OPERATION_ID = 1388058496
"""Id of the operation which is defined in the file (the value of the shape files written by NPVE)."""

_NO_OPERATION_ID = -1
"""Operation id of shapes of step and repeat files, which do not define operations."""


def _default_operation() -> t.Dict[str, t.Any]:
    """The operation ``Default`` of the shape files (all values as they are written by NPVE)."""
    return {
        'ID': _OPERATION_ID,
        'SortID': -1,
        'Name': 'Default',
        'Category': None,
        'Description': None,
        'ScanStyle': 'Standard',
        'StepCount': 1,
        'GasID': -1,
        'GasDelay': 0,
        'GasMinStartPressure': 0,
        'GasMinOpPressure': 0,
        'FloodGunMode': 0,
        'DitheringMode': 0,
        'VerificationColor': 536870911,
        'Model': {
            'MillSettings': {
                'EtchRate': 0,
                'Dose': '1.0',
                'Du': '0.5*SpotSize',
                'Dv': '0.5*SpotSize',
                'Dwell': '1.0',
                'HRefresh': 10,
                'VRefresh': 10,
                'Offset': 0,
                'Defocus': 'STD_DEFOCUS',
            },
        },
        'Apertures': None,
    }


class NPVESingleSite(NPVEBackendBase):
    """Exports a single site as NPVE shape file.

    In contrast to a step and repeat file (:class:`~.npve_step_and_repeat.NPVEStepAndRepeat`), the file contains neither
    a stage position nor a field of view, only the shapes relative to the center of the site. The layout may contain
    only one site with patterns; empty sites are ignored. Use :meth:`fibomat.layout.Layout.export_multi` to write one
    file per site.

    See :class:`~.backend_base.NPVEBackendBase` for the supported shapes, raster styles and deflection lists.
    """

    @check_units
    def __init__(
        self,
        description: t.Optional[str] = None,
        approximation_error: t.Optional[DimFloat[LengthDimension]] = None,
        max_embedded_points: int = DEFAULT_MAX_EMBEDDED_POINTS,
        system_name: str = 'WIN7ORION',
        beam: str = 'GFIS',
        aperture: str = '[1] He 20 µm (25.0kV|s=1.0)',
    ):
        """
        Args:
            description (str, optional): description
            approximation_error (DimFloat, optional): maximum distance between the arcs of hollow arc splines and
                their polygonal approximation, default ``0.0025 * unit('µm')``.
            max_embedded_points (int): deflection lists with more points are written to separate binary files. Use 0
                to write all lists to separate files.
            system_name (str): name of the system in the header of the file
            beam (str): beam in the header of the file
            aperture (str): name of the active aperture in the header of the file

        Raises:
            DimensionError: Raised if approximation_error is no dimensioned value or has not the dimension [length].
            ValueError: Raised if approximation_error is not positive or max_embedded_points is negative.
        """
        super().__init__(description, approximation_error, max_embedded_points)

        self._system_name = str(system_name)
        self._beam = str(beam)
        self._aperture = str(aperture)

        self._has_site = False
        self._shapes: t.List[FIBShape] = []

    def process_site(self, new_site: Site) -> None:
        if new_site.empty:
            return
        if self._has_site:
            raise ValueError(
                'NPVESingleSite supports only one site with patterns. Use NPVEStepAndRepeat or Layout.export_multi.'
            )
        self._has_site = True

        super().process_site(new_site)

    def _store_shape(self, shape: FIBShape) -> None:
        if not self._has_site:
            raise RuntimeError('Patterns must be added to a site.')

        # the shape files do not know "no operation" and need an angle
        if shape.mill.operation_id == _NO_OPERATION_ID:
            shape.mill.operation_id = _OPERATION_ID
        if shape.angle is None:
            shape.angle = 0.0

        self._shapes.append(shape)

    def _to_str(self) -> str:
        """The shape file as XML string.

        Raises:
            RuntimeError: Raised if there is no shape.
        """
        if not self._shapes:
            raise RuntimeError('The shape file does not contain any shapes.')

        document = {
            'nps': {
                'Version': '3.2',
                'Publisher': 'Fibics Incorporated',
                'Type': 'Nano Pattern Shape',
                'Extension': 'NPS',
                'MicroscopePersonality': {
                    'BeamController': {'Info': {'SystemName': self._system_name, 'Beam': self._beam}},
                    'Apertures': None,
                    'Operations': {'Operation': _default_operation()},
                },
                'ActiveAperture': {'ID': -1, 'Name': self._aperture},
                'FIBShapes': {'FIBShape': FIBShapeSchema(many=True).dump(self._shapes)},
            }
        }
        return xmltodict.unparse(document, encoding='iso-8859-1', pretty=True)
