# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat import Sample, Vector, U_
from fibomat import default_backends, shapes, curve_tools

sample = Sample()

spline = shapes.ArcSpline(
    [
        [ 7.50000000e-02,  4.33012702e-02, -5.77350269e-01],
        [ 1.38777878e-17,  8.66025404e-02, -5.77350269e-01],
        [-7.50000000e-02,  4.33012702e-02, -5.77350269e-01],
        [-7.50000000e-02, -4.33012702e-02, -5.77350269e-01],
        [-4.16333634e-17, -8.66025404e-02, -5.77350269e-01],
        [ 7.50000000e-02, -4.33012702e-02, -5.77350269e-01]
    ],
    is_closed=True
).scaled(10)

# smooth spline with a radius of 0.01
spline_smoothed = curve_tools.smooth(spline, .01)

unsmoothed_site = sample.create_site(
    dim_position=(-1.5, 0) * U_('µm'),
    dim_fov=(2, 2) * U_('µm'),
    description='unsmoothed'
)

unsmoothed_site.create_pattern(
    dim_shape=spline * U_('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

smoothed_site = sample.create_site(
    dim_position=(1.5, 0) * U_('µm'),
    dim_fov=(2, 2) * U_('µm'),
    description='smoothed'
)

smoothed_site.create_pattern(
    dim_shape=spline_smoothed * U_('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
