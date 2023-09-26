# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat import Sample, U_, Mill, Q_
from fibomat.layout import Group
from fibomat.shapes import Circle, Line
from fibomat import raster_styles
from fibomat.linalg import rotate, translate


sample = Sample()

marker = Group(
    [
        Line((-1, 0), (1, 0)),
        Line((0, -1), (0, 1)),
        Circle(r=.5, center=(0, 0)),
        Circle(r=.05, center=(.75, .75))
    ],
)

marker.pivot = lambda self: self.elements[2].center

single_marker_site = sample.create_site(
    dim_position=(0, 0) * U_('µm'),
    dim_fov=(2.5, 2.5) * U_('µm')
)

single_marker_site.create_pattern(
    dim_shape=marker * U_('µm'),
    mill=Mill(dwell_time=Q_('1 ms'), repeats=1),
    raster_style=raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
)

# generate four copies and translate and rotate them
first_marker = marker.translated((4, 4))
second_marker = marker.transformed(translate((-4, 4)) | rotate(np.pi/2, origin='pivot'))
third_marker = marker.transformed(translate((-4, -4)) | rotate(np.pi, origin='pivot'))
fourth_marker = marker.transformed(translate((4, -4)) | rotate(3*np.pi/2, origin='pivot'))


corner_marker_site = sample.create_site(
    dim_position=(8, 0) * U_('µm'),
    dim_fov=(10, 10) * U_('µm')
)

for corner_marker in [first_marker, second_marker, third_marker, fourth_marker]:
    corner_marker_site.create_pattern(
        dim_shape=corner_marker * U_('µm'),
        mill=Mill(dwell_time=Q_('1 ms'), repeats=1),
        raster_style=raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
