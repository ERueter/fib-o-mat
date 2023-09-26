# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from fibomat import Sample, Vector, U_
from fibomat import default_backends, shapes, curve_tools

sample = Sample()

site = sample.create_site(
    dim_position=(0, 0) * U_('µm'),
    dim_fov=(5, 5) * U_('µm'),
    description='curve intersections'
)

circle = shapes.Circle(r=2.4)

# deflate the circle a few times and then rasterize it

for defl_rect in curve_tools.deflate(circle.to_arc_spline(), 0.5):
    rasterized = curve_tools.rasterize(defl_rect, 0.25)

    for point in rasterized.positions:
        site.create_pattern(
            dim_shape=shapes.Spot(point) * U_('µm'),
            mill=None,
            raster_style=default_backends.StubRasterStyle(1)
        )

    site.create_pattern(
        dim_shape=defl_rect * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
