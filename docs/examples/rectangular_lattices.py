# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from typing import Tuple

import numpy as np

from fibomat import Sample, U_, Mill, Q_, Pattern, Site
from fibomat.layout import Group, DimGroup, Lattice, DimLattice
from fibomat.shapes import Circle, Line, Spot, Rect, Ellipse
from fibomat import raster_styles
from fibomat.default_backends import StubRasterStyle


def ellipse_generator(pos_xy: Tuple[float, float], pos_uv: Tuple[int, int]):
    ellipse_base_radius = .1
    u, v = pos_uv

    # the position of the shape does not matter here.
    return Ellipse(a=ellipse_base_radius + 0.05 * u, b=ellipse_base_radius + 0.05 * v)


def checkerboad_generator(pos_xy: Tuple[float, float], pos_uv: Tuple[int, int]):
    size = .2
    u, v = pos_uv

    if (u + v) % 2 == 0:
        return Rect(width=size, height=size)
    else:
        return None


sample = Sample()

spot_lattice_site = sample.create_site(
    dim_position=(-2.5, 0) * U_('µm'),
    dim_fov=(2, 2) * U_('µm')
)

spot_lattice = Lattice.generate_rect(nu=3, nv=4, du=.75, dv=.5, element=Spot((0, 0)))  # the position of the spot does not matter

spot_lattice_site.create_pattern(
    dim_shape=spot_lattice*U_('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

# ----------------------------------------------------------------------------------------------------------------------

ellipse_lattice_site = sample.create_site(
    dim_position=(0, 0) * U_('µm'),
    dim_fov=(2, 2) * U_('µm')
)

checkerboard_lattice = Lattice.generate_rect(nu=4, nv=4, du=.5, dv=.5, element=ellipse_generator, center=(-.15/2, .15/2))

ellipse_lattice_site.create_pattern(
    dim_shape=checkerboard_lattice*U_('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

# ----------------------------------------------------------------------------------------------------------------------

checkerboard_lattice_site = sample.create_site(
    dim_position=(2.5, 0) * U_('µm'),
    dim_fov=(2, 2) * U_('µm')
)

checkerboard_lattice = Lattice.generate_rect(nu=8, nv=8, du=.2, dv=.2, element=checkerboad_generator)

checkerboard_lattice_site.create_pattern(
    dim_shape=checkerboard_lattice*U_('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=2)
)


sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))

#
# non_square_lattice_site = sample.create_site(
#     dim_position=(0.5, 0) * U_('µm'),
#     dim_fov=(7, 7) * U_('µm')
# )
#
# non_square_lattice = Lattice.generate(
#     boundary=Rect(4, 4),
#     u=(np.sqrt(5)/2, 0), v=(0, -np.sqrt(5)/2),
#     element=Circle(r=.25),
# ).rotated(np.pi/4 - np.deg2rad(60))
#
# non_square_lattice_site.create_pattern(
#     dim_shape=non_square_lattice * U_('µm'),
#     mill=None,
#     raster_style=raster_styles.two_d.LineByLine(
#         line_pitch=Q_('1 nm'),
#         scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
#         alpha=0,
#         invert=False,
#         line_style=raster_styles.one_d.Curve(
#             pitch=Q_('1 nm'),scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
#         )
#     )
# )
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# square_pattern_site = sample.create_site(
#     dim_position=(8.5, 0) * U_('µm'),
#     dim_fov=(7, 7) * U_('µm')
# )
#
# marker = Group([
#     Line((-1, 0), (1, 0)),
#     Line((0, -1), (0, 1)),
#     Circle(r=.5, center=(0, 0)),
#     Circle(r=.05, center=(.75, .75))
# ]).scaled(.75)
#
# marker.pivot = lambda self: self._elements[2].center
#
# pattern = Pattern(
#     dim_shape=marker * U_('µm'),
#     mill=Mill(dwell_time=Q_('1 ms'), repeats=1),
#     raster_style=raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
# )
#
# square_pattern_lattice = DimLattice.generate_rect(
#     nu=3, nv=3,
#     dim_du=2 * U_('µm'), dim_dv=2 * U_('µm'),
#     dim_element=pattern
# )
#
# square_pattern_site.add_pattern(square_pattern_lattice)
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# site_lattice = DimLattice.generate_rect(
#     nu=4, nv=4,
#     dim_du=4 * U_('µm'), dim_dv=4 * U_('µm'),
#     dim_element=Site(dim_center=(0, 0) * U_('µm'), dim_fov=(3, 3) * U_('µm')),
#     dim_center=(4.5, -12) * U_('µm')
# )
#
# # site_lattice[0, 1].add_pattern(pattern)
# # site_lattice[2, 3].add_pattern(pattern)
#
# sample.add_site(site_lattice)
#
# sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
