# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat import Sample, U_, Mill, Q_, Pattern
from fibomat.layout import Group, DimGroup
from fibomat.shapes import Circle, Line, Spot
from fibomat import raster_styles
from fibomat.linalg import rotate, translate



sample = Sample()

center_group = Group([
    Line((-1, 0), (1, 0)),
    Line((0, -1), (0, 1)),
    Circle(r=.5, center=(0, 0))
])

marker_pattern = DimGroup([
    Pattern(
        dim_shape=center_group * U_('µm'),
        mill=Mill(dwell_time=Q_('1 ms'), repeats=1),
        raster_style=raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    ),
    Pattern(
        dim_shape=Spot((.75, .75)) * U_('µm'),
        mill=Mill(dwell_time=Q_('1 ms'), repeats=1),
        raster_style=raster_styles.zero_d.SingleSpot()
    )
])

# we chose again the center of circle located at third element of the  of the first pattern
marker_pattern.pivot = lambda self: self.elements[0].dim_shape.elements[2].shape.center * self.elements[0].dim_shape.elements[2].unit


single_marker_site = sample.create_site(
    dim_position=(0, 0) * U_('µm'),
    dim_fov=(2.5, 2.5) * U_('µm')
)

single_marker_site += marker_pattern


# generate four copies and translate and rotate them
first_marker_pattern = marker_pattern.translated((4, 4) * U_('µm'))
second_marker_pattern = marker_pattern.transformed(translate((-4, 4) * U_('µm')) | rotate(np.pi/2, origin='pivot'))
third_marker_pattern = marker_pattern.transformed(translate((-4, -4) * U_('µm')) | rotate(np.pi, origin='pivot'))
fourth_marker_pattern = marker_pattern.transformed(translate((4, -4) * U_('µm')) | rotate(3*np.pi/2, origin='pivot'))

corner_marker_site = sample.create_site(
    dim_position=(8, 0) * U_('µm'),
    dim_fov=(10, 10) * U_('µm')
)

for corner_marker_pattern in [first_marker_pattern, second_marker_pattern, third_marker_pattern, fourth_marker_pattern]:
    corner_marker_site += corner_marker_pattern

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
