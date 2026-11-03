"""Provides the :class:`NPVEStepAndRepeat` backend, which exports a layout as NPVE step and repeat file (XML).

Example::

    mill = NPVEMill(dwell_time=1. * unit('µs'), repeats=2)
    exported = layout.export(NPVEStepAndRepeat)
    exported.save('patterns.xml')

.. note:: The file format is defined by the schemas in ``common_schemas.py`` and ``sar_schemas.py``; the tests compare
    the generated files with reference files, so that the format does not change by accident.
"""
from __future__ import annotations

import typing as t
import warnings

import xmltodict

from fibomat.default_backends.npve.step_and_repeat.backend_base import (
    DEFAULT_MAX_EMBEDDED_POINTS, NPVEBackendBase
)
from fibomat.default_backends.npve.step_and_repeat.common_models import FIBShape
from fibomat.default_backends.npve.step_and_repeat.sar_models import SaRFile, SaRSharedShapes, SaRSite
from fibomat.default_backends.npve.step_and_repeat.sar_schemas import SaRFileSchema
from fibomat.layout.site import Site
from fibomat.units import DimFloat, LengthDimension, check_units, unit


__all__ = ['NPVEStepAndRepeat', 'StepAndRepeatBackend']


_MICRON = unit('µm')


class NPVEStepAndRepeat(NPVEBackendBase):
    """Exports a layout as NPVE step and repeat file: a list of sites with stage moves.

    See :class:`~.backend_base.NPVEBackendBase` for the supported shapes, raster styles and deflection lists. Use
    :class:`~.npve_single_site.NPVESingleSite` for a shape file of a single site.
    """

    _first_shape_id = 10000

    @check_units
    def __init__(
        self,
        share_patterns: bool = False,
        skip_empty_sites: bool = True,
        description: t.Optional[str] = None,
        approximation_error: t.Optional[DimFloat[LengthDimension]] = None,
        max_embedded_points: int = DEFAULT_MAX_EMBEDDED_POINTS,
        spot_size: t.Optional[DimFloat[LengthDimension]] = None,
    ):
        """
        Args:
            share_patterns (bool): if True, only the first site contains patterns, which are shared by all sites (the
                other sites must be empty).
            skip_empty_sites (bool): if True, empty sites are ignored, otherwise they raise an error (NPVE does not do
                what everybody expects if an empty site is in a step and repeat list). Ignored (False) if
                `share_patterns` is True.
            description (str, optional): description
            approximation_error (DimFloat, optional): maximum distance between the arcs of hollow arc splines and
                their polygonal approximation, default ``0.0025 * unit('µm')``.
            max_embedded_points (int): deflection lists with more points are written to separate binary files. Use 0
                to write all lists to separate files.

            spot_size (DimFloat, optional): spot size of the beam (as shown by NPVE for the beam which is used for
                patterning). NPVE converts the doses of spots and lines with the spot size of its active beam, so spots
                and lines with a dose need it; mills with repeats do not. For a spot with a dose, NPVE ignores the dwell
                time of the mill and calculates it from the dose and the beam current.

        Raises:
            DimensionError: Raised if approximation_error or spot_size is no dimensioned value or has not the dimension
                [length].
            ValueError: Raised if approximation_error or spot_size is not positive or max_embedded_points is negative.
        """
        super().__init__(description, approximation_error, max_embedded_points, spot_size)

        self._share_patterns = bool(share_patterns)
        self._shared_patterns: t.Union[SaRSharedShapes, t.List[t.Any]] = []

        self._skip_empty_sites = False if self._share_patterns else bool(skip_empty_sites)

        self._site_index = 0

        self._current_site: t.Optional[SaRSite] = None
        self._sites: t.List[SaRSite] = []

    # sites

    def process_site(self, new_site: Site) -> None:
        if self._share_patterns:
            if self._site_index != 0 and not new_site.empty:
                raise ValueError("Only first site should contain patterns.")
        elif new_site.empty:
            if self._skip_empty_sites:
                return
            raise ValueError(
                "Site may not be empty (NPVE is not doing what everybody would expect "
                "if an empty site is encountered in an step and repeat list)."
            )

        if self._sites:
            last_site_pos = self._sites[-1].center
        else:
            last_site_pos = new_site.center.vector_as(_MICRON)

        self._current_site = SaRSite(self._site_index, new_site, last_site_pos)
        self._sites.append(self._current_site)
        self._site_index += 1

        if self._site_index == 1 or not self._share_patterns:
            super().process_site(new_site)

            if self._share_patterns:
                self._shared_patterns = SaRSharedShapes(self._current_site.shapes["shapes_list"])
                self._current_site.shapes["shapes_list"] = []

    def _store_shape(self, shape: FIBShape) -> None:
        if self._current_site is None:
            raise RuntimeError('Patterns must be added to a site.')
        self._current_site.add_fib_shape(shape)

    def _to_str(self) -> str:
        """The step and repeat file as XML string.

        Raises:
            RuntimeError: Raised if there are no sites.
        """
        if not self._sites:
            raise RuntimeError("Step and repeat list does not contain any sites.")

        if self._share_patterns:
            sar_file = SaRFile(self._sites, self._shared_patterns)
        else:
            sar_file = SaRFile(self._sites)
        return xmltodict.unparse(SaRFileSchema().dump(sar_file), encoding="iso-8859-1", pretty=True)


class StepAndRepeatBackend(NPVEStepAndRepeat):
    """Deprecated name of :class:`NPVEStepAndRepeat`."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        """
        Args:
            *args: see :class:`NPVEStepAndRepeat`
            **kwargs: see :class:`NPVEStepAndRepeat`
        """
        warnings.warn(
            'StepAndRepeatBackend is deprecated and will be removed in a future version, use NPVEStepAndRepeat.',
            DeprecationWarning, stacklevel=2
        )
        super().__init__(*args, **kwargs)
