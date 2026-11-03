"""Provides :class:`NPVEBackendBase`, the common part of the backends which write NPVE shapes.

The base class maps the shapes of the library to NPVE shapes (:class:`~.common_models.FIBShape`) and writes deflection
lists. The derived classes :class:`~.npve_step_and_repeat.NPVEStepAndRepeat` and
:class:`~.npve_single_site.NPVESingleSite` only define where the shapes are stored and how the file looks like.

Deflection lists are patterns with the shape :class:`~fibomat.shapes.RasterizedPoints` and the raster style
:class:`~fibomat.raster_styles.zero_d.PreRasterized`. The dwell time of a point is
``mill.dwell_time * point weight``, the weights must be positive integers. Short lists are embedded in the file; lists
with more than `max_embedded_points` points are written to separate binary files next to the file (see
:mod:`~.deflection_list`).
"""
from __future__ import annotations

import abc
import pathlib
import typing as t
import warnings

import numpy as np

from fibomat.backend import BackendBase
from fibomat.composite_shapes import HollowArcSpline, Ring
from fibomat.curve_tools.rasterize import rasterize_with_const_error
from fibomat.default_backends.npve.step_and_repeat.common_models import FIBShape, ShapeTexture
from fibomat.default_backends.npve.step_and_repeat.deflection_list import DeflectionList
from fibomat.default_backends.npve.step_and_repeat.npve_mill import scan_direction_of
from fibomat.mill import DoseMill
from fibomat.layout.pattern import Pattern
from fibomat.linalg import Vector
from fibomat.shapes import Circle, DimShape, Ellipse, Line, Polygon, Polyline, RasterizedPoints, Rect, Spot
from fibomat.units import (
    AreaDoseDimension, DimensionError, DimFloat, LengthDimension, LineDoseDimension, SpotDoseDimension, check_units,
    scale_factor, unit
)
from fibomat.utils import PathLike


__all__ = ['NPVEBackendBase', 'DEFAULT_MAX_EMBEDDED_POINTS']


_MICRON = unit('µm')

_NODE_FIRST = 0
_NODE_NEXT = 1
_NODE_LAST = 129
"""NPVE node types: first node, further nodes and last node (which closes the shape)."""

_DOSE_OF_RASTER_DIMENSION = {
    0: (SpotDoseDimension, 'spot dose (e.g. ions)'),
    1: (LineDoseDimension, 'line dose (e.g. nC / µm)'),
    2: (AreaDoseDimension, 'area dose (e.g. ions / nm**2)'),
}
"""The dose which NPVE expects for each dimension of the raster style."""

DEFAULT_MAX_EMBEDDED_POINTS = 5000
"""Deflection lists with more points are written to a separate binary file instead of being embedded.

An embedded point needs about 150 bytes in the file (12 bytes in a binary file), so this limit corresponds to a file
size of about 750 kB per deflection list.
"""


class NPVEBackendBase(BackendBase, abc.ABC):
    """Base class of the backends which write NPVE shapes.

    All patterns must use an :class:`~fibomat.default_backends.npve.NPVEMill` and the raster styles
    :class:`~fibomat.raster_styles.zero_d.SingleSpot`, :class:`~fibomat.raster_styles.one_d.Curve` (scan sequence
    consecutive), :class:`~fibomat.raster_styles.two_d.LineByLine` (also
    :class:`~fibomat.default_backends.npve.LineByLineOutlined`) or, for deflection lists,
    :class:`~fibomat.raster_styles.zero_d.PreRasterized`. Supported shapes are spots, lines, polylines, polygons,
    rectangles, circles, ellipses (not rotated), rings, hollow arc splines (which are approximated by polygons) and
    rasterized points (deflection lists).

    Patterns with the keyword argument ``use_bitmap`` are exported as bitmap texture (this needs the bitmap backend).
    """

    _first_shape_id = 1
    """Display id of the first shape."""

    @check_units
    def __init__(
        self,
        description: t.Optional[str] = None,
        approximation_error: t.Optional[DimFloat[LengthDimension]] = None,
        max_embedded_points: int = DEFAULT_MAX_EMBEDDED_POINTS,
    ):
        """
        Args:
            description (str, optional): description
            approximation_error (DimFloat, optional): maximum distance between the arcs of hollow arc splines and
                their polygonal approximation, default ``0.0025 * unit('µm')``.
            max_embedded_points (int): deflection lists with more points are written to separate binary files. Use 0
                to write all lists to separate files.

        Raises:
            DimensionError: Raised if approximation_error is no dimensioned value or has not the dimension [length].
            ValueError: Raised if approximation_error is not positive or max_embedded_points is negative.
        """
        super().__init__(description)

        if approximation_error is None:
            approximation_error = 0.0025 * _MICRON
        if not approximation_error.magnitude > 0.:
            raise ValueError('approximation_error must be positive.')
        self._approximation_error = approximation_error.m_as('µm')

        if max_embedded_points < 0:
            raise ValueError('max_embedded_points must not be negative.')
        self._max_embedded_points = max_embedded_points

        self._warned_about_spot_dose = False
        self._shape_index = self._first_shape_id
        self._deflection_lists: t.List[DeflectionList] = []

    @abc.abstractmethod
    def _store_shape(self, shape: FIBShape) -> None:
        """Store a finished shape.

        Args:
            shape (FIBShape): shape
        """

    @abc.abstractmethod
    def _to_str(self) -> str:
        """The file as XML string."""

    def process_pattern(self, ptn: Pattern) -> None:
        if "use_bitmap" in ptn.kwargs:
            self._add_bitmap(ptn)
        else:
            super().process_pattern(ptn)

    def validate_pattern(self, ptn: Pattern) -> None:
        """Check that the dose of the mill fits to the raster style and to the shape.

        The dose must be a spot dose for raster styles of dimension 0, a line dose for dimension 1 and an area dose for
        dimension 2. Deflection lists (rasterized points) cannot have a dose. The other requirements of NPVE (mill type,
        raster styles and scan sequences) are checked when the shape is created.

        Args:
            ptn (Pattern): pattern to be checked.

        Raises:
            DimensionError: Raised if the dose does not fit to the raster style.
            ValueError: Raised if a deflection list has a dose.
        """
        if not isinstance(ptn.mill, DoseMill) or ptn.mill.dose is None:
            return

        name = f'Pattern {ptn.description!r}' if ptn.description else 'A pattern'

        if isinstance(ptn.dim_shape.shape, RasterizedPoints):
            raise ValueError(f'{name} is a deflection list. Deflection lists need a mill with repeats, a dose is not '
                             'supported.')

        dimension = ptn.raster_style.dimension
        expected = _DOSE_OF_RASTER_DIMENSION.get(dimension)
        if expected is not None and not expected[0].matches(ptn.mill.dose.quantity):
            raise DimensionError(
                f'{name} has the raster style {type(ptn.raster_style).__name__} (dimension {dimension}), which needs a '
                f'{expected[1]}, but the mill has the dose {ptn.mill.dose.units:~P}.'
            )

    def _add_bitmap(self, ptn: Pattern) -> None:
        """Add a pattern as bitmap texture of a rectangle (this needs the bitmap backend)."""
        self.validate_pattern(ptn)

        from fibomat.default_backends.bitmap_backend import BitmapBackend  # pylint: disable=import-outside-toplevel
        from fibomat.layout import Layout  # pylint: disable=import-outside-toplevel

        bbox = ptn.bounding_box
        layout = Layout()
        site = layout.create_site(dim_position=bbox.center, dim_fov=(bbox.width, bbox.height))
        site += ptn
        bitmap = layout.export(BitmapBackend).image()

        texture = ShapeTexture(bitmap, (bbox.width, bbox.height), ptn.raster_style)

        half_width = bbox.width.m_as("µm") / 2
        half_height = bbox.height.m_as("µm") / 2
        center = Vector(bbox.center[0].m_as("µm"), bbox.center[1].m_as("µm"))

        self._add_shape(
            "TRectangle", center,
            [
                (center + (-half_width, -half_height), _NODE_FIRST),
                (center + (half_width, -half_height), _NODE_NEXT),
                (center + (half_width, half_height), _NODE_NEXT),
                (center + (-half_width, half_height), _NODE_LAST),
            ],
            ptn, shape_texture=texture
        )

    # file

    def print(self) -> None:
        """Print the XML of the file."""
        print(self._to_str())

    def save(self, filename: PathLike) -> None:
        """Save the file and, next to it, the binary files of the deflection lists which are not embedded.

        The binary files are called ``<stem of filename>_<id of the shape>.dat``.

        Args:
            filename (PathLike): filename

        Raises:
            RuntimeError: Raised if the file would be empty.
        """
        path = pathlib.Path(filename)
        for deflection_list in self._deflection_lists:
            deflection_list.set_file_stem(path.stem)

        text = self._to_str()
        with open(path, "w", encoding="iso-8859-1") as file:
            file.write(text)

        for deflection_list in self._deflection_lists:
            if not deflection_list.embedded:
                path.with_name(deflection_list.data_filename).write_bytes(deflection_list.to_bytes())

    # shapes

    def _add_shape(
        self, class_: str, rotation_center: t.Any, nodes: t.List[t.Tuple[t.Any, int]], ptn: Pattern, **kwargs: t.Any
    ) -> None:
        """Add a shape (coordinates in µm)."""
        self._store_shape(
            FIBShape(
                class_=class_, id=self._shape_index, rotation_center=rotation_center, nodes=nodes, mill=ptn.mill,
                raster_style=ptn.raster_style, **kwargs
            )
        )
        self._shape_index += 1

    def rasterized_points(self, ptn: Pattern[RasterizedPoints]) -> None:
        """Add a deflection list.

        Raises:
            ValueError: Raised if a weight is no positive integer.
            TypeError: Raised if the mill is no NPVEMill or the raster style is no PreRasterized.
        """
        points = np.array(ptn.dim_shape.shape.dwell_points, dtype=float)
        deflection_list = DeflectionList(
            positions=points[:, :2] * scale_factor(_MICRON, ptn.dim_shape.unit), repeats=points[:, 2],
            shape_id=self._shape_index, embedded=len(points) <= self._max_embedded_points,
        )
        self._add_shape(
            "TDeflectionListShape", (0., 0.), [], ptn, angle=0.0, deflection_list=deflection_list,
            shape_name=f"Deflection List {self._shape_index}",
        )
        self._deflection_lists.append(deflection_list)

    @staticmethod
    def _scaled(ptn: Pattern) -> t.Any:
        """The shape of a pattern in µm."""
        return ptn.dim_shape.shape.scaled(scale_factor(_MICRON, ptn.dim_shape.unit))

    @staticmethod
    def _derived(ptn: Pattern, shape: t.Any, **kwargs: t.Any) -> Pattern:
        """A pattern with another shape (in the unit of the pattern) and the mill and raster style of `ptn`."""
        return Pattern(DimShape(shape, ptn.dim_shape.unit), ptn.mill, ptn.raster_style, **kwargs)

    def spot(self, ptn: Pattern[Spot]) -> None:
        if not self._warned_about_spot_dose:
            warnings.warn("Check the spot dose in NPVE as the calculation includes some magic numbers.", stacklevel=2)
            self._warned_about_spot_dose = True

        scaled_spot = self._scaled(ptn)
        self._add_shape("TFIBSpot", scaled_spot.center, [(scaled_spot.center, _NODE_FIRST)], ptn)

    def rect(self, ptn: Pattern[Rect]) -> None:
        rect: Rect = ptn.dim_shape.shape

        if ptn.raster_style.dimension == 1:
            self.polyline(self._derived(ptn, Polyline([*rect.corners, rect.corners[0]])))
        else:
            self.polygon(self._derived(
                ptn, Polygon(rect.corners, description=rect.description), description=ptn.description, **ptn.kwargs
            ))

    def polygon(self, ptn: Pattern[Polygon]) -> None:
        if ptn.raster_style.dimension == 1:
            points = ptn.dim_shape.shape.points
            self.polyline(self._derived(ptn, Polyline([*points, points[0]])))
        else:
            scaled_poly: Polygon = self._scaled(ptn)

            nodes = [(scaled_poly.points[0], _NODE_FIRST)]
            nodes.extend((point, _NODE_NEXT) for point in scaled_poly.points[1:-1])
            nodes.append((scaled_poly.points[-1], _NODE_LAST))

            self._add_shape("TPolygon", scaled_poly.center, nodes, ptn)

    def polyline(self, ptn: Pattern[Polyline]) -> None:
        is_line = len(ptn.dim_shape.shape.points) == 2
        scaled_poly: Polyline = self._scaled(ptn)

        nodes = [(scaled_poly.points[0], _NODE_FIRST)]

        if is_line:
            nodes.append((scaled_poly.points[1], _NODE_LAST))
        else:
            nodes.extend((point, _NODE_NEXT) for point in scaled_poly.points[1:])

        self._add_shape("TLine" if is_line else "TPolyline", scaled_poly.center, nodes, ptn)

    def line(self, ptn: Pattern[Line]) -> None:
        line: Line = ptn.dim_shape.shape
        self.polyline(self._derived(ptn, Polyline([line.start, line.end])))

    def circle(self, ptn: Pattern[Circle]) -> None:
        scaled_circle: Circle = self._scaled(ptn)
        r = scaled_circle.r  # pylint: disable=invalid-name
        center = scaled_circle.center

        self._add_shape(
            "TEllipse", center,
            [
                (center + (-r, -r), _NODE_FIRST),
                (center + (r, -r), _NODE_NEXT),
                (center + (r, r), _NODE_NEXT),
                (center + (-r, r), _NODE_LAST),
            ],
            ptn
        )

    def ellipse(self, ptn: Pattern[Ellipse]) -> None:
        scaled_ellipse: Ellipse = self._scaled(ptn)

        if abs(scaled_ellipse.theta) > 0.0001:
            raise NotImplementedError("Rotated ellipses are not supported by the step and repeat backend.")

        a, b = scaled_ellipse.a, scaled_ellipse.b  # pylint: disable=invalid-name
        center = scaled_ellipse.center

        self._add_shape(
            "TEllipse", center,
            [
                (center + (-a, -b), _NODE_FIRST),
                (center + (a, -b), _NODE_NEXT),
                (center + (a, b), _NODE_NEXT),
                (center + (-a, b), _NODE_LAST),
            ],
            ptn
        )

    def ring(self, ptn: Pattern[Ring]) -> None:
        scaled_ring: Ring = self._scaled(ptn)
        r_outer = scaled_ring.r_outer
        center = scaled_ring.center

        scan_direction = scan_direction_of(ptn.mill)
        self._add_shape(
            "TRing", center,
            [
                (center + (-r_outer, r_outer), _NODE_FIRST),
                (center + (r_outer, r_outer), _NODE_NEXT),
                (center + (r_outer, -r_outer), _NODE_NEXT),
                (center + (-r_outer, -r_outer), _NODE_LAST),
            ],
            ptn, outline_thickness=scaled_ring.thickness, angle=0.0 if scan_direction is None else scan_direction
        )

    def hollow_arc_spline(self, ptn: Pattern[HollowArcSpline]) -> None:
        def make_node_list(points: np.ndarray) -> t.List[t.Tuple[t.Any, int]]:
            if len(points) < 3:
                raise ValueError("A polygon needs at least three points.")

            nodes_ = [(points[0], _NODE_FIRST)]
            nodes_.extend((point, _NODE_NEXT) for point in points[1:-1])
            nodes_.append((points[-1], _NODE_LAST))
            return nodes_

        scaled_hollow_arc_spline: HollowArcSpline = self._scaled(ptn)

        nodes: t.List[t.Tuple[t.Any, int]] = []
        for arc_spline in [scaled_hollow_arc_spline.boundary] + list(scaled_hollow_arc_spline.holes):
            nodes.extend(make_node_list(rasterize_with_const_error(arc_spline, self._approximation_error).points))

        self._add_shape("TPolygon", scaled_hollow_arc_spline.center, nodes, ptn)
