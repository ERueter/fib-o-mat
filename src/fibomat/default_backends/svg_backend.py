"""Provides the :class:`SVGBackend`, which draws a layout as scalable vector graphic.

Every site is a layer (an Inkscape layer, so the sites can be shown and hidden in Inkscape). Areas (raster styles of
dimension 2) are drawn filled, curves as lines and spots as small dots. Arcs are exact SVG arcs; only parametric
curves are approximated (by arc splines).

Optionally, the fields of view of the sites are drawn and a mill setting (dwell time, number of repeats or dose) is
encoded by a color map of matplotlib, with a color bar next to the drawing.

Example::

    from fibomat.default_backends.svg_backend import SVGBackend

    layout.export(SVGBackend).save('layout.svg')

    # color the shapes by their dwell time, without fields of view
    layout.export(SVGBackend, color_by='dwell_time', colormap='plasma', show_fov=False).save('dwell_times.svg')
"""
from __future__ import annotations

import io
import math
import typing as t
import xml.etree.ElementTree as ET

import numpy as np
import pint  # type: ignore

from fibomat import composite_shapes, shapes
from fibomat.backend import BackendBase
from fibomat.default_backends.color_by import ColorBy
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import Site
from fibomat.linalg import BoundingBox, DimBoundingBox, Vector
from fibomat.shapes import DimShape
from fibomat.units import DimFloat, LengthDimension, check_units, scale_to, unit
from fibomat.utils import PathLike


__all__ = ['SVGBackend']


_SVG_NAMESPACE = 'http://www.w3.org/2000/svg'
_INKSCAPE_NAMESPACE = 'http://www.inkscape.org/namespaces/inkscape'

ET.register_namespace('', _SVG_NAMESPACE)
ET.register_namespace('inkscape', _INKSCAPE_NAMESPACE)

_DEFAULT_PIXEL_SIZE = 0.01 * unit('µm')

_LABELS = {ColorBy.DWELL_TIME: 'dwell time', ColorBy.REPEATS: 'repeats', ColorBy.DOSE: 'dose'}
"""Labels of the color bars."""

_N_GRADIENT_STOPS = 32
"""Number of colors which are sampled from the color map for a color bar."""


def _svg_tag(name: str) -> str:
    return f'{{{_SVG_NAMESPACE}}}{name}'


def _fmt(value: float) -> str:
    """Format a coordinate for the svg file (at most 4 decimal places, no trailing zeros)."""
    text = f'{value:.4f}'.rstrip('0').rstrip('.')
    return '0' if text in ('-0', '') else text


class _Loop(t.NamedTuple):
    """A curve of an svg path: vertices ``(x, y, bulge)`` in pixels (y up) and if it is closed."""

    vertices: np.ndarray
    closed: bool


class _Element(t.NamedTuple):
    """A drawn shape, in pixels with the y axis pointing up (it is flipped when the svg file is created)."""

    loops: t.List[_Loop]
    """Curves of a path (empty for a spot)."""
    spot: t.Optional[Vector]
    """Center of a spot (a dot of radius `spot_radius`)."""
    filled: bool
    bounding_box: BoundingBox
    value: t.Optional[t.Tuple[str, float]]
    """Color bar key and value of the encoded mill setting (None: no value)."""
    description: t.Optional[str]


class _Layer(t.NamedTuple):
    """The elements of a site."""

    label: str
    fov: t.Optional[BoundingBox]
    elements: t.List[_Element]


class _ColorScale(t.NamedTuple):
    """Range of the values of one color bar (one per unit dimensionality, e.g. spot, line and area doses)."""

    label: str
    unit: t.Optional[pint.Unit]
    minimum: float
    maximum: float

    def normalize(self, value: float) -> float:
        """Map a value to [0, 1] (0.5 if all values are equal)."""
        if self.maximum == self.minimum:
            return 0.5
        return (value - self.minimum) / (self.maximum - self.minimum)


class SVGBackend(BackendBase):
    """Draws a layout as svg file.

    The svg file has one layer per site (Inkscape layers). Shapes whose raster style has dimension 2 are drawn filled
    (if they are closed), all other shapes as lines and spots as dots. The drawing is in pixels of the size
    `pixel_size`: a length of ``100 * pixel_size`` in the layout is 100 pixels long in the svg file.

    If `color_by` is given, the mill setting ``mill['dwell_time']``, ``mill['repeats']`` or ``mill['dose']`` of each
    pattern is encoded by a matplotlib color map (patterns whose mill does not have the setting are drawn in
    `missing_color`). Values with different dimensions (e.g. spot doses and area doses) get separate color scales.

    Groups, lattices and texts are split into their shapes, so every shape is drawn.
    """

    @check_units
    def __init__(
        self,
        *,
        pixel_size: t.Optional[DimFloat[LengthDimension]] = None,
        stroke_width: t.Optional[DimFloat[LengthDimension]] = None,
        spot_radius: t.Optional[DimFloat[LengthDimension]] = None,
        show_fov: bool = True,
        color_by: t.Optional[t.Union[ColorBy, str]] = None,
        colormap: t.Any = 'viridis',
        color_bar: bool = True,
        color: str = 'black',
        missing_color: str = 'lightgrey',
        fov_color: str = 'grey',
        description: t.Optional[str] = None,
    ):
        """
        Args:
            pixel_size (DimFloat, optional): length of one pixel (svg user unit), default ``0.01 * unit('µm')``.
            stroke_width (DimFloat, optional): width of the lines of curves and fields of view. Default: 0.2 % of the
                larger side of the drawing.
            spot_radius (DimFloat, optional): radius of the dots of spots. Default: two stroke widths.
            show_fov (bool): if True, the fields of view of the sites are drawn as dashed rectangles.
            color_by (ColorBy, str, optional): mill setting which is encoded by color: ``'dwell_time'``,
                ``'repeats'`` or ``'dose'``. If None, all shapes have the color `color`.
            colormap (str, matplotlib.colors.Colormap): matplotlib color map (name or object) used with `color_by`.
            color_bar (bool): if True and `color_by` is given, color bars with the value ranges are drawn right of the
                layout.
            color (str): color of the shapes if `color_by` is None (any svg color).
            missing_color (str): color of the shapes without the setting `color_by`.
            fov_color (str): color of the fields of view.
            description (str, optional): description, used as title of the svg file.

        Raises:
            DimensionError: Raised if a length is no dimensioned value or has not the dimension [length].
            ValueError: Raised if a length is not positive or `color_by` is unknown.
            ImportError: Raised if `color_by` is given and matplotlib is not installed.
        """
        super().__init__(description)

        self._pixel_size = pixel_size if pixel_size is not None else _DEFAULT_PIXEL_SIZE
        for name, length in (('pixel_size', self._pixel_size), ('stroke_width', stroke_width),
                             ('spot_radius', spot_radius)):
            if length is not None and not (math.isfinite(length.magnitude) and length.magnitude > 0.):
                raise ValueError(f'{name} must be positive and finite, got {length!r}.')

        self._stroke_width = self._to_pixels(stroke_width) if stroke_width is not None else None
        self._spot_radius = self._to_pixels(spot_radius) if spot_radius is not None else None

        self._show_fov = bool(show_fov)
        self._color_by = ColorBy.parse(color_by) if color_by is not None else None
        self._colormap = self._load_colormap(colormap) if self._color_by is not None else None
        self._color_bar = bool(color_bar)
        self._color = str(color)
        self._missing_color = str(missing_color)
        self._fov_color = str(fov_color)

        self._layers: t.List[_Layer] = []
        self._site_center = Vector(0, 0)

        # unit of each color scale (keyed by the dimensionality of the values)
        self._value_units: t.Dict[str, t.Optional[pint.Unit]] = {}

    @staticmethod
    def _load_colormap(colormap: t.Any) -> t.Any:
        """Get a matplotlib color map by name (or check that `colormap` is one).

        Raises:
            ImportError: Raised if matplotlib is not installed.
            ValueError: Raised if the color map is unknown.
        """
        try:
            import matplotlib  # pylint: disable=import-outside-toplevel
            import matplotlib.colors  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as error:
            raise ImportError(
                'Color maps need matplotlib; install fibomat with the "exporting" extra '
                '(pip install fibomat[exporting]).'
            ) from error

        if isinstance(colormap, matplotlib.colors.Colormap):
            return colormap
        try:
            return matplotlib.colormaps[colormap]
        except KeyError:
            raise ValueError(f'Unknown matplotlib color map {colormap!r}.') from None

    def _to_pixels(self, length: DimFloat[LengthDimension]) -> float:
        return length.m_as(self._pixel_size.units) / self._pixel_size.magnitude

    def _pixel_factor(self, length_unit: t.Any) -> float:
        """Factor which converts lengths in `length_unit` to pixels."""
        return 1. / scale_to(length_unit, self._pixel_size)

    # ------------------------------------------------------------------------------------------------------------------
    # sites and patterns
    # ------------------------------------------------------------------------------------------------------------------

    def process_site(self, new_site: Site) -> None:
        center = new_site.center
        self._site_center = center.vector_as(self._pixel_size.units) / self._pixel_size.magnitude

        fov = None
        if self._show_fov:
            try:
                fov = self._dim_bounding_box_to_pixels(new_site.fov_bounding_box)
            except ValueError:
                pass  # an empty site without explicit fov has no fov

        label = new_site.description or f'Site {len(self._layers)}'
        self._layers.append(_Layer(label=label, fov=fov, elements=[]))

        super().process_site(new_site)

    def _dim_bounding_box_to_pixels(self, bounding_box: DimBoundingBox) -> BoundingBox:
        lower_left = bounding_box.lower_left.vector_as(self._pixel_size.units) / self._pixel_size.magnitude
        upper_right = bounding_box.upper_right.vector_as(self._pixel_size.units) / self._pixel_size.magnitude
        return BoundingBox(lower_left, upper_right)

    def _current_layer(self) -> _Layer:
        if not self._layers:
            raise RuntimeError('Patterns must be added to a site.')
        return self._layers[-1]

    def _value(self, ptn: Pattern) -> t.Optional[t.Tuple[str, float]]:
        """The color bar key and the value of the mill setting `color_by` of a pattern (None if it has none)."""
        if self._color_by is None or ptn.mill is None:
            return None

        try:
            value = ptn.mill[self._color_by.value]
        except KeyError:
            return None

        if isinstance(value, DimFloat):
            value = value.quantity

        if isinstance(value, pint.Quantity):
            key = str(value.dimensionality)
            if key not in self._value_units:
                self._value_units[key] = value.units
            return key, float(value.m_as(self._value_units[key]))

        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            self._value_units.setdefault('', None)
            return '', float(value)

        raise TypeError(
            f'The mill setting {self._color_by.value!r} must be a number or a dimensioned value, got {value!r}.'
        )

    def _transform(self, ptn: Pattern) -> t.Tuple[float, Vector]:
        """Scale factor (shape unit -> pixels) and offset (site center in pixels) of a pattern."""
        return self._pixel_factor(ptn.dim_shape.unit), self._site_center

    def _add(self, ptn: Pattern, loops: t.List[_Loop], filled: bool, bounding_box: BoundingBox,
             spot: t.Optional[Vector] = None) -> None:
        self._current_layer().elements.append(_Element(
            loops=loops, spot=spot, filled=filled, bounding_box=bounding_box, value=self._value(ptn),
            description=ptn.description,
        ))

    def _arc_spline_loop(self, curve: shapes.ArcSpline, factor: float, offset: Vector) -> _Loop:
        vertices = np.array(curve.vertices, dtype=float)
        vertices[:, :2] = vertices[:, :2] * factor + np.asarray(offset)
        return _Loop(vertices, curve.is_closed)

    def _shape_bounding_box(self, bounding_box: BoundingBox, factor: float, offset: Vector) -> BoundingBox:
        return BoundingBox(bounding_box.lower_left * factor + offset, bounding_box.upper_right * factor + offset)

    def _draw_curve(self, ptn: Pattern, curve: shapes.ArcSpline) -> None:
        """Draw a curve: filled if it is closed and the raster style fills areas, otherwise as line."""
        factor, offset = self._transform(ptn)
        raster_dimension = getattr(ptn.raster_style, 'dimension', 1)
        filled = curve.is_closed and raster_dimension == 2
        self._add(
            ptn, [self._arc_spline_loop(curve, factor, offset)], filled,
            self._shape_bounding_box(curve.bounding_box, factor, offset)
        )

    def _draw_shape(self, ptn: Pattern) -> None:
        self._draw_curve(ptn, ptn.dim_shape.shape.to_arc_spline())

    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        factor, offset = self._transform(ptn)
        center = ptn.dim_shape.shape.center * factor + offset
        self._add(ptn, [], True, BoundingBox(center, center), spot=center)

    def rasterized_points(self, ptn: Pattern[shapes.RasterizedPoints]) -> None:
        # every dwell point is drawn as spot
        for point in ptn.dim_shape.shape.dwell_points[:, :2]:
            self.spot(Pattern(
                DimShape(shapes.Spot(point), ptn.dim_shape.unit), ptn.mill, ptn.raster_style,
                description=ptn.description, **ptn.kwargs
            ))

    def line(self, ptn: Pattern[shapes.Line]) -> None:
        self._draw_shape(ptn)

    def polyline(self, ptn: Pattern[shapes.Polyline]) -> None:
        self._draw_shape(ptn)

    def polygon(self, ptn: Pattern[shapes.Polygon]) -> None:
        self._draw_shape(ptn)

    def rect(self, ptn: Pattern[shapes.Rect]) -> None:
        self._draw_shape(ptn)

    def circle(self, ptn: Pattern[shapes.Circle]) -> None:
        self._draw_shape(ptn)

    def ellipse(self, ptn: Pattern[shapes.Ellipse]) -> None:
        self._draw_shape(ptn)

    def arc(self, ptn: Pattern[shapes.Arc]) -> None:
        self._draw_shape(ptn)

    def arc_spline(self, ptn: Pattern[shapes.ArcSpline]) -> None:
        self._draw_shape(ptn)

    def parametric_curve(self, ptn: Pattern[shapes.ParametricCurve]) -> None:
        # approximated by an arc spline
        self._draw_shape(ptn)

    def hollow_arc_spline(self, ptn: Pattern[composite_shapes.HollowArcSpline]) -> None:
        factor, offset = self._transform(ptn)
        hollow = ptn.dim_shape.shape
        loops = [self._arc_spline_loop(hollow.boundary, factor, offset)]
        loops += [self._arc_spline_loop(hole, factor, offset) for hole in hollow.holes]
        filled = getattr(ptn.raster_style, 'dimension', 2) == 2
        self._add(ptn, loops, filled, self._shape_bounding_box(hollow.boundary.bounding_box, factor, offset))

    def ring(self, ptn: Pattern[composite_shapes.Ring]) -> None:
        self.hollow_arc_spline(Pattern(
            DimShape(ptn.dim_shape.shape.to_hollow_arc_spline(), ptn.dim_shape.unit), ptn.mill, ptn.raster_style,
            description=ptn.description, **ptn.kwargs
        ))

    def process_unknown(self, ptn: Pattern) -> None:
        """Draw custom shapes which can be converted to (hollow) arc splines; raise for all others."""
        shape = ptn.dim_shape.shape
        if callable(getattr(shape, 'to_hollow_arc_spline', None)):
            self.ring(ptn)
        elif callable(getattr(shape, 'to_arc_spline', None)):
            self._draw_shape(ptn)
        else:
            super().process_unknown(ptn)

    # ------------------------------------------------------------------------------------------------------------------
    # svg output
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def color_scales(self) -> t.List[t.Tuple[str, t.Optional[pint.Unit], float, float]]:
        """The color scales ``(label, unit, minimum, maximum)`` (empty if `color_by` is None).

        Access:
            get
        """
        return [tuple(scale) for scale in self._color_scales().values()]  # type: ignore[misc]

    def _color_scales(self) -> t.Dict[str, _ColorScale]:
        if self._color_by is None:
            return {}

        values: t.Dict[str, t.List[float]] = {}
        for layer in self._layers:
            for element in layer.elements:
                if element.value is not None:
                    values.setdefault(element.value[0], []).append(element.value[1])

        return {
            key: _ColorScale(_LABELS[self._color_by], self._value_units[key], min(group), max(group))
            for key, group in values.items()
        }

    def _element_color(self, element: _Element, scales: t.Dict[str, _ColorScale]) -> str:
        if self._color_by is None:
            return self._color
        if element.value is None:
            return self._missing_color
        key, value = element.value
        return self._hex_color(scales[key].normalize(value))

    def _hex_color(self, normalized: float) -> str:
        import matplotlib.colors  # pylint: disable=import-outside-toplevel
        return str(matplotlib.colors.to_hex(self._colormap(normalized)))

    def _drawing_bounding_box(self) -> BoundingBox:
        boxes = [element.bounding_box for layer in self._layers for element in layer.elements]
        boxes += [layer.fov for layer in self._layers if layer.fov is not None]
        if not boxes:
            raise RuntimeError('The layout does not contain any pattern.')

        total = boxes[0]
        for box in boxes[1:]:
            total = total.extended(box)
        return total

    def to_element(self) -> ET.Element:
        """Create the svg document.

        Returns:
            xml.etree.ElementTree.Element: the ``<svg>`` element

        Raises:
            RuntimeError: Raised if the layout does not contain any pattern.
        """
        # pylint: disable=too-many-locals
        drawing_box = self._drawing_bounding_box()
        size = max(drawing_box.width, drawing_box.height)
        if size <= 0.:
            size = 100.  # a single spot

        stroke_width = self._stroke_width if self._stroke_width is not None else 0.002 * size
        spot_radius = self._spot_radius if self._spot_radius is not None else 2. * stroke_width
        margin = max(0.02 * size, spot_radius + stroke_width)

        left = drawing_box.lower_left.x - margin
        top = drawing_box.upper_right.y + margin
        width = drawing_box.width + 2 * margin
        height = drawing_box.height + 2 * margin

        def to_svg(x: float, y: float) -> str:
            # svg coordinates: origin at the upper left corner, y axis pointing down
            return f'{_fmt(x - left)},{_fmt(top - y)}'

        scales = self._color_scales()
        draw_color_bars = self._color_bar and bool(scales)
        font_size = 0.03 * height
        bar_block_width = 0.06 * height + 7 * font_size
        total_width = width + (len(scales) * bar_block_width if draw_color_bars else 0.)

        svg = ET.Element(_svg_tag('svg'), {
            'version': '1.1',
            'width': _fmt(total_width), 'height': _fmt(height),
            'viewBox': f'0 0 {_fmt(total_width)} {_fmt(height)}',
        })
        if self.description:
            ET.SubElement(svg, _svg_tag('title')).text = self.description

        for layer in self._layers:
            group = ET.SubElement(svg, _svg_tag('g'), {
                f'{{{_INKSCAPE_NAMESPACE}}}groupmode': 'layer', f'{{{_INKSCAPE_NAMESPACE}}}label': layer.label,
            })

            for element in layer.elements:
                color = self._element_color(element, scales)
                if element.spot is not None:
                    node = ET.SubElement(group, _svg_tag('circle'), {
                        'cx': _fmt(element.spot.x - left), 'cy': _fmt(top - element.spot.y),
                        'r': _fmt(spot_radius), 'fill': color,
                    })
                else:
                    attributes = {'d': self._path_data(element.loops, to_svg)}
                    if element.filled:
                        attributes.update({'fill': color, 'fill-rule': 'evenodd', 'stroke': 'none'})
                    else:
                        attributes.update({
                            'fill': 'none', 'stroke': color, 'stroke-width': _fmt(stroke_width),
                            'stroke-linejoin': 'round', 'stroke-linecap': 'round',
                        })
                    node = ET.SubElement(group, _svg_tag('path'), attributes)
                if element.description:
                    ET.SubElement(node, _svg_tag('title')).text = element.description

            if layer.fov is not None:
                ET.SubElement(group, _svg_tag('rect'), {
                    'x': _fmt(layer.fov.lower_left.x - left), 'y': _fmt(top - layer.fov.upper_right.y),
                    'width': _fmt(layer.fov.width), 'height': _fmt(layer.fov.height),
                    'fill': 'none', 'stroke': self._fov_color, 'stroke-width': _fmt(stroke_width),
                    'stroke-dasharray': f'{_fmt(4 * stroke_width)},{_fmt(2 * stroke_width)}',
                })

        if draw_color_bars:
            self._add_color_bars(svg, list(scales.values()), width, height, font_size)

        return svg

    @staticmethod
    def _path_data(loops: t.List[_Loop], to_svg: t.Callable[[float, float], str]) -> str:
        """Svg path data of curves of lines and arcs.

        The bulge of a segment is ``tan(sweep / 4)``; positive values are counterclockwise arcs. The y axis of the
        svg file points down, so the angles of the svg file grow clockwise on the screen (sweep flag 1). The flip of
        the y axis keeps the orientation on the screen: counterclockwise arcs stay counterclockwise and get the sweep
        flag 0.
        """
        commands = []
        for loop in loops:
            vertices = loop.vertices
            n_vertices = len(vertices)
            commands.append(f'M {to_svg(vertices[0, 0], vertices[0, 1])}')

            n_segments = n_vertices if loop.closed else n_vertices - 1
            for i in range(n_segments):
                x_start, y_start, bulge = vertices[i]
                x_end, y_end = vertices[(i + 1) % n_vertices, :2]
                if bulge == 0.:
                    if not (loop.closed and i == n_segments - 1):  # the last line of a closed curve is drawn by 'Z'
                        commands.append(f'L {to_svg(x_end, y_end)}')
                else:
                    chord = math.hypot(x_end - x_start, y_end - y_start)
                    radius = chord * (1. + bulge ** 2) / (4. * abs(bulge))
                    large_arc = 1 if abs(bulge) > 1. else 0
                    sweep = 0 if bulge > 0. else 1
                    commands.append(f'A {_fmt(radius)},{_fmt(radius)} 0 {large_arc} {sweep} {to_svg(x_end, y_end)}')

            if loop.closed:
                commands.append('Z')
        return ' '.join(commands)

    def _add_color_bars(
        self, svg: ET.Element, scales: t.List[_ColorScale], x_start: float, height: float, font_size: float
    ) -> None:
        """Add a vertical color bar for each color scale right of the drawing."""
        defs = ET.SubElement(svg, _svg_tag('defs'))
        bar_height = 0.6 * height
        bar_width = 0.04 * height
        bar_top = 0.2 * height
        block_width = 0.06 * height + 7 * font_size

        for i_scale, scale in enumerate(scales):
            gradient_id = f'fibomat-color-bar-{i_scale}'
            gradient = ET.SubElement(defs, _svg_tag('linearGradient'), {
                'id': gradient_id, 'x1': '0', 'y1': '1', 'x2': '0', 'y2': '0',
            })
            for i_stop in range(_N_GRADIENT_STOPS):
                position = i_stop / (_N_GRADIENT_STOPS - 1)
                ET.SubElement(gradient, _svg_tag('stop'), {
                    'offset': _fmt(position), 'stop-color': self._hex_color(position),
                })

            x_bar = x_start + 0.02 * height + i_scale * block_width
            group = ET.SubElement(svg, _svg_tag('g'), {
                f'{{{_INKSCAPE_NAMESPACE}}}groupmode': 'layer', f'{{{_INKSCAPE_NAMESPACE}}}label': 'Color bar',
                'font-family': 'sans-serif', 'font-size': _fmt(font_size),
            })
            ET.SubElement(group, _svg_tag('rect'), {
                'x': _fmt(x_bar), 'y': _fmt(bar_top), 'width': _fmt(bar_width), 'height': _fmt(bar_height),
                'fill': f'url(#{gradient_id})', 'stroke': 'black', 'stroke-width': _fmt(0.05 * font_size),
            })

            label = scale.label if scale.unit is None else f'{scale.label} / {scale.unit:~P}'
            ET.SubElement(group, _svg_tag('text'), {
                'x': _fmt(x_bar), 'y': _fmt(bar_top - 0.8 * font_size),
            }).text = label

            for value, y_position in ((scale.maximum, bar_top), (scale.minimum, bar_top + bar_height)):
                ET.SubElement(group, _svg_tag('text'), {
                    'x': _fmt(x_bar + bar_width + 0.3 * font_size), 'y': _fmt(y_position + 0.35 * font_size),
                }).text = f'{value:.3g}'

    def to_string(self) -> str:
        """The svg document as string.

        Returns:
            str
        """
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(self.to_element(), encoding='unicode')

    def save(self, filename: PathLike) -> None:
        """Save the svg file.

        Args:
            filename (PathLike): file name, should end with ``.svg``
        """
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(self.to_string())

    def to_pillow_image(self, background_color: t.Tuple[int, int, int, int] = (255, 255, 255, 255)) -> t.Any:
        """Render the svg file to a pillow image (needs the packages cairosvg and pillow).

        Args:
            background_color (Tuple[int, int, int, int]): RGBA background color

        Returns:
            PIL.Image.Image: RGB image

        Raises:
            ImportError: Raised if cairosvg or pillow is not installed.
        """
        try:
            import cairosvg  # type: ignore  # pylint: disable=import-outside-toplevel
            import PIL.Image  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as error:
            raise ImportError(
                'Rendering the svg file needs cairosvg and pillow (pip install cairosvg pillow).'
            ) from error

        rendered = PIL.Image.open(io.BytesIO(cairosvg.svg2png(bytestring=self.to_string().encode('utf-8'))))
        image = PIL.Image.new('RGBA', size=rendered.size, color=background_color)
        image.paste(rendered, None, rendered)
        return image.convert('RGB')
