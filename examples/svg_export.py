"""Export a layout as svg file; the dwell times are encoded by a color map."""
from fibomat import shapes
from fibomat.arrangements import Lattice
from fibomat.composite_shapes import Ring
from fibomat.default_backends import SVGBackend
from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d, two_d
from fibomat.units import unit


um = unit('µm')
curve = one_d.Curve(10 * unit('nm'), ScanSequence.CONSECUTIVE)
area = two_d.LineByLine.make_equal_pitch(10 * unit('nm'))

layout = Layout(description='SVG export')

site = layout.create_site(Vector(0, 0) * um, Vector(10, 10) * um, description='shapes')
site.create_pattern(shapes.Rect(3, 1.5, theta=0.3, center=(-2, 2)) * um, Mill(1 * unit('ms'), 1), area)
site.create_pattern(shapes.Circle(1.2, center=(2, 2)) * um, Mill(2 * unit('ms'), 1), area)
site.create_pattern(Ring(1.3, 0.5, center=(-2, -2)) * um, Mill(3 * unit('ms'), 1), area)
site.create_pattern(shapes.Arc.from_points((1, -2.5), (2, -1.5), (3, -2.5)) * um, Mill(4 * unit('ms'), 1), curve)

# the dwell time of the lattice elements grows with the index of the lattice point
lattice_site = layout.create_site(Vector(12, 0) * um, description='lattice')
for i in range(3):
    lattice_site.create_pattern(
        Lattice.from_counts(3, 1, 2, 2, shapes.Circle(0.6)).translated((0, 2 - 2 * i)) * um,
        Mill((i + 1) * 1.5 * unit('ms'), 1), area
    )

layout.export(SVGBackend, color_by='dwell_time', colormap='viridis').save('svg_export.svg')
