# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from fibomat import Sample, Vector, U_
from fibomat import default_backends, shapes, curve_tools

sample = Sample()

c1 = shapes.ArcSpline.from_shape(shapes.Circle(r=1, center=(-.75, 0)))
c2 = shapes.ArcSpline.from_shape(shapes.Circle(r=1, center=(.75, 0)))


# Union
union_site = sample.create_site(
    dim_position=(-2.5, 2.5) * U_('µm'),
    dim_fov=(4, 4) * U_('µm'),
    description='Union'
)

union_curves = curve_tools.combine_curves(c1, c2, 'union')
for curve in union_curves['remaining']:
    union_site.create_pattern(
        dim_shape=curve * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

# Xor
xor_site = sample.create_site(
    dim_position=(2.5, 2.5) * U_('µm'),
    dim_fov=(4, 4) * U_('µm'),
    description='Xor'
)

xor_curves = curve_tools.combine_curves(c1, c2, 'xor')
for curve in xor_curves['remaining']:
    xor_site.create_pattern(
        dim_shape=curve * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

# Exclude
exclude_site = sample.create_site(
    dim_position=(-2.5, -2.5) * U_('µm'),
    dim_fov=(4, 4) * U_('µm'),
    description='Exclude'
)

exclude_curves = curve_tools.combine_curves(c1, c2, 'exclude')
for curve in exclude_curves['remaining']:
    exclude_site.create_pattern(
        dim_shape=curve * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

# Intersect
intersect_site = sample.create_site(
    dim_position=(2.5, -2.5) * U_('µm'),
    dim_fov=(4, 4) * U_('µm'),
    description='Intersect'
)

intersect_curves = curve_tools.combine_curves(c1, c2, 'intersect')
for curve in intersect_curves['remaining']:
    intersect_site.create_pattern(
        dim_shape=curve * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
