# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from fibomat import Sample, Vector, U_
from fibomat import default_backends, shapes, curve_tools

sample = Sample()

# curve intersections
spline = shapes.ArcSpline(
    [(-.6, 0, -.2), (-.4, 1., -.2), (-.2, 0., .2), (0, 1, .2), (.2, 0, -.2), (.4, 1, -.2), (.6, 0, 0)],
    is_closed=False
).translated((0., -.5))

lines = [
    shapes.ArcSpline.from_shape(shapes.Line(start=(-.8, .25), end=(.8, .25))),
    shapes.ArcSpline.from_shape(shapes.Line(start=(-.8, -.25), end=(.8, -.25)))
]

curve_intersection_site = sample.create_site(
    dim_position=(-1, 0) * U_('µm'),
    dim_fov=(1.8, 1.8) * U_('µm'),
    description='curve intersections'
)

curve_intersection_site.create_pattern(
    dim_shape=spline * U_('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

for line in lines:
    curve_intersection_site.create_pattern(
        dim_shape=line * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )

    intersections = curve_tools.curve_intersections(spline, line)

    for intersection in intersections['intersections']:
        curve_intersection_site.create_pattern(
            dim_shape=shapes.Spot(intersection['pos']) * U_('µm'),
            mill=None,
            raster_style=default_backends.StubRasterStyle(1)
        )

# self_intersections
fish = shapes.ArcSpline.from_segments([
    shapes.Arc.from_points((.1, -.5), (.4, 0), (1.3, 0)),
    shapes.Arc.from_points((1.3, 0), (.5, 0), (.1, .5))
])
fish = fish.translated(-fish.center)

self_intersection_site = sample.create_site(
    dim_position=(1, 0) * U_('µm'),
    dim_fov=(1.8, 1.8) * U_('µm'),
    description='self intersections'
)

self_intersection_site.create_pattern(
    dim_shape=fish * U_('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

self_intersections = curve_tools.self_intersections(fish)

for intersection in self_intersections['intersections']:
    self_intersection_site.create_pattern(
        dim_shape=shapes.Spot(intersection['pos']) * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )


sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
