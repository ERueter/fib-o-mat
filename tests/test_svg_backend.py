"""Tests of `fibomat.default_backends.svg_backend`."""
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fibomat.arrangements import Lattice
from fibomat.composite_shapes import Ring
from fibomat.default_backends import ColorBy, SVGBackend
from fibomat.default_backends.npve import NPVEMill
from fibomat.layout import Layout
from fibomat.linalg import DimVector
from fibomat.mill import Mill, SpecialMill
from fibomat.raster_styles import ScanSequence, one_d, two_d, zero_d
from fibomat.shapes import Arc, Circle, Line, Polygon, RasterizedPoints, Rect, Spot
from fibomat.units import DimensionError, unit

matplotlib = pytest.importorskip('matplotlib')


SVG = '{http://www.w3.org/2000/svg}'
INKSCAPE = '{http://www.inkscape.org/namespaces/inkscape}'


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def curve():
    return one_d.Curve(10 * unit('nm'), ScanSequence.CONSECUTIVE)


def area():
    return two_d.LineByLine.make_equal_pitch(10 * unit('nm'))


def mill(dwell_ms=1., repeats=1):
    return Mill(dwell_ms * unit('ms'), repeats)


def export(layout, **kwargs):
    backend = layout.export(SVGBackend, **kwargs)
    return backend, ET.fromstring(backend.to_string().split('\n', 1)[1])


def layers(svg):
    return [g for g in svg.findall(f'{SVG}g') if g.get(f'{INKSCAPE}label') != 'Color bar']


def shapes_of(layer):
    return [child for child in layer if child.tag in (f'{SVG}path', f'{SVG}circle')]


def fov_rects(svg):
    return [rect for layer in layers(svg) for rect in layer.findall(f'{SVG}rect')]


def two_site_layout():
    layout = Layout(description='design')
    site = layout.create_site(um(0, 0), um(10, 10), description='first')
    site.create_pattern(Rect(2, 1) * unit('µm'), mill(1.), area(), description='rect')
    site.create_pattern(Line((0, 0), (1, 0)) * unit('µm'), mill(3.), curve())
    other = layout.create_site(um(20, 0), description='second')
    other.create_pattern(Spot((0, 0)) * unit('µm'), mill(2.), zero_d.SingleSpot())
    return layout


class TestStructure:
    def test_layers_and_shapes(self):
        _, svg = export(two_site_layout())
        assert svg.find(f'{SVG}title').text == 'design'
        assert [layer.get(f'{INKSCAPE}label') for layer in layers(svg)] == ['first', 'second']

        rect, line = shapes_of(layers(svg)[0])
        assert rect.get('fill') == 'black' and rect.get('stroke') == 'none'
        assert rect.find(f'{SVG}title').text == 'rect'
        assert line.get('fill') == 'none' and line.get('stroke') == 'black'

        (spot,) = shapes_of(layers(svg)[1])
        assert spot.tag == f'{SVG}circle' and spot.get('fill') == 'black'

    def test_unnamed_sites_are_numbered(self):
        layout = Layout()
        for x in (0, 10):
            layout.create_site(um(x, 0)).create_pattern(Spot() * unit('µm'), mill(), zero_d.SingleSpot())
        _, svg = export(layout)
        assert [layer.get(f'{INKSCAPE}label') for layer in layers(svg)] == ['Site 0', 'Site 1']

    def test_coordinates(self):
        # 1 pixel = 10 nm (default); the y axis points down in the svg file
        layout = Layout()
        site = layout.create_site(um(1, 2))
        site.create_pattern(Line((0, 0), (1, 1)) * unit('µm'), mill(), curve())
        _, svg = export(layout, show_fov=False, stroke_width=10 * unit('nm'))

        (line,) = shapes_of(layers(svg)[0])
        start, end = (tuple(map(float, point.split(','))) for point in re.findall(r'[-\d.]+,[-\d.]+', line.get('d')))
        assert end[0] - start[0] == pytest.approx(100)
        assert end[1] - start[1] == pytest.approx(-100)
        assert line.get('stroke-width') == '1'

    def test_pixel_size(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Line((0, 0), (1, 0)) * unit('µm'), mill(), curve())
        small, _ = export(layout, show_fov=False)
        large, _ = export(layout, show_fov=False, pixel_size=1 * unit('nm'))
        width = lambda backend: float(ET.fromstring(backend.to_string().split('\n', 1)[1]).get('width'))
        assert width(large) == pytest.approx(10 * width(small))

    def test_closed_curve_and_arcs(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(Rect(2, 1) * unit('µm'), mill(), area())
        # upper half circle from (1, 0) to (-1, 0): counterclockwise -> sweep flag 0 (y axis flipped)
        site.create_pattern(Arc.from_points((1, 0), (0, 1), (-1, 0)) * unit('µm'), mill(), curve())
        _, svg = export(layout, show_fov=False)

        rect, arc = shapes_of(layers(svg)[0])
        assert rect.get('d').endswith('Z')
        assert rect.get('d').count('L') == 3  # the fourth side is drawn by 'Z'
        match = re.search(r'A ([\d.]+),[\d.]+ 0 (\d) (\d)', arc.get('d'))
        assert float(match.group(1)) == pytest.approx(100)
        assert match.group(2) == '0' and match.group(3) == '0'

    def test_filled_only_with_area_raster_style(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(Circle(1) * unit('µm'), mill(), curve())
        site.create_pattern(Polygon([(0, 0), (1, 0), (0, 1)]) * unit('µm'), mill(), area())
        _, svg = export(layout)
        outline, filled = shapes_of(layers(svg)[0])
        assert outline.get('fill') == 'none'
        assert filled.get('fill') == 'black'

    def test_ring_has_a_hole(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Ring(2, 0.5) * unit('µm'), mill(), area())
        _, svg = export(layout)
        (ring,) = shapes_of(layers(svg)[0])
        assert ring.get('fill-rule') == 'evenodd'
        assert ring.get('d').count('M') == 2

    def test_arrangements_and_rasterized_points(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(Lattice.from_counts(2, 3, 2, 2, Circle(0.5)) * unit('µm'), mill(), area())
        points = RasterizedPoints(np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]], dtype=float), False)
        site.create_pattern(points * unit('µm'), mill(), zero_d.PreRasterized())
        _, svg = export(layout)
        elements = shapes_of(layers(svg)[0])
        assert len(elements) == 6 + 3
        assert sum(element.tag == f'{SVG}circle' for element in elements) == 3

    def test_spot_radius(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Spot() * unit('µm'), mill(), zero_d.SingleSpot())
        _, svg = export(layout, spot_radius=50 * unit('nm'))
        (spot,) = shapes_of(layers(svg)[0])
        assert spot.get('r') == '5'

    def test_empty_layout(self):
        backend = Layout().export(SVGBackend)
        with pytest.raises(RuntimeError, match='any pattern'):
            backend.to_string()

    def test_save(self, tmp_path):
        backend = two_site_layout().export(SVGBackend)
        backend.save(tmp_path / 'layout.svg')
        content = (tmp_path / 'layout.svg').read_text(encoding='utf-8')
        assert content.startswith('<?xml') and '</svg>' in content


class TestFov:
    def test_shown_by_default(self):
        _, svg = export(two_site_layout())
        # the second site contains a single spot and has no explicit fov, so it has no fov (no extent)
        assert [len(layer.findall(f'{SVG}rect')) for layer in layers(svg)] == [1, 0]
        rects = fov_rects(svg)
        assert all(rect.get('fill') == 'none' and rect.get('stroke-dasharray') for rect in rects)
        # the first site has an explicit fov of 10 µm = 1000 pixels
        assert float(rects[0].get('width')) == pytest.approx(1000)

    def test_hidden(self):
        _, svg = export(two_site_layout(), show_fov=False)
        assert fov_rects(svg) == []

    def test_fov_extends_the_drawing(self):
        _, with_fov = export(two_site_layout())
        _, without_fov = export(two_site_layout(), show_fov=False)
        assert float(with_fov.get('height')) > float(without_fov.get('height'))

    def test_fov_color(self):
        _, svg = export(two_site_layout(), fov_color='red')
        assert {rect.get('stroke') for rect in fov_rects(svg)} == {'red'}

    def test_automatic_fov(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Rect(2, 1) * unit('µm'), mill(), area())
        _, svg = export(layout)
        (rect,) = fov_rects(svg)
        # smallest square around the center which contains the pattern, scaled with the default fov scale
        assert float(rect.get('width')) == pytest.approx(float(rect.get('height')))
        assert float(rect.get('width')) >= 200


class TestColorMap:
    def test_dwell_time(self):
        backend, svg = export(two_site_layout(), color_by='dwell_time', colormap='viridis')
        viridis = matplotlib.colormaps['viridis']

        rect, line = shapes_of(layers(svg)[0])
        (spot,) = shapes_of(layers(svg)[1])
        assert rect.get('fill') == matplotlib.colors.to_hex(viridis(0.))
        assert spot.get('fill') == matplotlib.colors.to_hex(viridis(0.5))
        assert line.get('stroke') == matplotlib.colors.to_hex(viridis(1.))

        ((label, value_unit, low, high),) = backend.color_scales
        assert label == 'dwell time' and value_unit == (1 * unit('ms')).units
        assert (low, high) == (1., 3.)

    def test_values_are_converted_to_one_unit(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(Spot() * unit('µm'), Mill(1 * unit('ms'), 1), zero_d.SingleSpot())
        site.create_pattern(Spot((1, 0)) * unit('µm'), Mill(500 * unit('µs'), 1), zero_d.SingleSpot())
        backend, _ = export(layout, color_by=ColorBy.DWELL_TIME)
        ((_, _, low, high),) = backend.color_scales
        assert (low, high) == pytest.approx((0.5, 1.))

    def test_repeats(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        for i, repeats in enumerate((1, 5, 3)):
            site.create_pattern(Spot((i, 0)) * unit('µm'), mill(repeats=repeats), zero_d.SingleSpot())
        backend, _ = export(layout, color_by='repeats', colormap='plasma')
        assert backend.color_scales == [('repeats', None, 1., 5.)]

    def test_dose_with_different_dimensions(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(
            Spot() * unit('µm'), NPVEMill(dwell_time=1 * unit('µs'), dose=100 * unit('ions')), zero_d.SingleSpot()
        )
        site.create_pattern(
            Rect(1, 1, center=(3, 0)) * unit('µm'), NPVEMill(dwell_time=1 * unit('µs'), dose=2 * unit('ions / nm**2')),
            area()
        )
        site.create_pattern(Rect(1, 1, center=(6, 0)) * unit('µm'), mill(), area(), description='no dose')
        backend, svg = export(layout, color_by='dose')

        assert len(backend.color_scales) == 2
        bars = [g for g in svg.findall(f'{SVG}g') if g.get(f'{INKSCAPE}label') == 'Color bar']
        assert len(bars) == 2
        labels = [bar.find(f'{SVG}text').text for bar in bars]
        assert labels[0].startswith('dose / ') and labels[0] != labels[1]

        # a mill without a dose is drawn in the missing color
        assert shapes_of(layers(svg)[0])[2].get('fill') == 'lightgrey'

    def test_missing_color(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(Line((0, 0), (1, 0)) * unit('µm'), SpecialMill(), curve())
        site.create_pattern(Line((0, 1), (1, 1)) * unit('µm'), mill(), curve())
        _, svg = export(layout, color_by='dwell_time', missing_color='red')
        assert shapes_of(layers(svg)[0])[0].get('stroke') == 'red'

    def test_color_bar(self):
        _, with_bar = export(two_site_layout(), color_by='dwell_time')
        _, without_bar = export(two_site_layout(), color_by='dwell_time', color_bar=False)
        _, no_colors = export(two_site_layout())

        bar_labels = lambda svg: [g for g in svg.findall(f'{SVG}g') if g.get(f'{INKSCAPE}label') == 'Color bar']
        assert len(bar_labels(with_bar)) == 1
        assert bar_labels(without_bar) == [] and bar_labels(no_colors) == []
        assert float(with_bar.get('width')) > float(without_bar.get('width'))

        texts = [text.text for text in bar_labels(with_bar)[0].findall(f'{SVG}text')]
        assert texts == ['dwell time / ms', '3', '1']
        assert len(with_bar.find(f'{SVG}defs').find(f'{SVG}linearGradient')) > 2

    def test_invalid_mill_setting(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(
            Line((0, 0), (1, 0)) * unit('µm'), SpecialMill(dwell_time='long'), curve()
        )
        with pytest.raises(TypeError, match='dwell_time'):
            layout.export(SVGBackend, color_by='dwell_time')

    def test_colormap_object(self):
        _, svg = export(two_site_layout(), color_by='dwell_time', colormap=matplotlib.colormaps['magma'])
        rect = shapes_of(layers(svg)[0])[0]
        assert rect.get('fill') == matplotlib.colors.to_hex(matplotlib.colormaps['magma'](0.))

    def test_single_value_is_in_the_middle(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Spot() * unit('µm'), mill(), zero_d.SingleSpot())
        _, svg = export(layout, color_by='dwell_time')
        (spot,) = shapes_of(layers(svg)[0])
        assert spot.get('fill') == matplotlib.colors.to_hex(matplotlib.colormaps['viridis'](0.5))

    def test_without_color_by_the_color_is_used(self):
        backend, svg = export(two_site_layout(), color='blue')
        assert backend.color_scales == []
        assert {element.get('fill') for element in shapes_of(layers(svg)[1])} == {'blue'}


class TestInvalidArguments:
    @pytest.mark.parametrize('name', ['pixel_size', 'stroke_width', 'spot_radius'])
    def test_lengths(self, name):
        with pytest.raises(DimensionError, match=name):
            SVGBackend(**{name: 1.})
        with pytest.raises(DimensionError, match=name):
            SVGBackend(**{name: 1. * unit('s')})
        with pytest.raises(ValueError, match=name):
            SVGBackend(**{name: 0. * unit('µm')})

    def test_color_by(self):
        with pytest.raises(ValueError, match='ColorBy'):
            SVGBackend(color_by='current')

    def test_colormap(self):
        with pytest.raises(ValueError, match='color map'):
            SVGBackend(color_by='dwell_time', colormap='no such map')

    def test_positional_arguments_are_not_allowed(self):
        with pytest.raises(TypeError):
            SVGBackend(1 * unit('nm'))


def test_pillow_image():
    pytest.importorskip('cairosvg')
    image = two_site_layout().export(SVGBackend).to_pillow_image()
    assert image.mode == 'RGB' and image.size[0] > 0


def test_pillow_image_without_cairosvg(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cairosvg', None)
    with pytest.raises(ImportError, match='cairosvg'):
        two_site_layout().export(SVGBackend).to_pillow_image()
