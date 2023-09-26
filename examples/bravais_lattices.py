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
from fibomat.linalg import Vector


sample = Sample()

kagome_lattice_site = sample.create_site(
    dim_position=(-3, 0) * U_('µm'),
    dim_fov=(5, 5) * U_('µm')
)

unit_cell = Group([Spot((-.25/2, 0)), Spot((.25/2, 0)), Spot((0, .25 * np.sqrt(3) / 2))])

kagome_lattice = Lattice.generate(boundary=Circle(5.1),  u=(.5, 0), v=Vector(.5, 0).rotated(np.pi/3), element=unit_cell)

kagome_lattice_site.create_pattern(
    dim_shape=kagome_lattice*U_('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

# ----------------------------------------------------------------------------------------------------------------------

kagome_lattice_fixed_site = sample.create_site(
    dim_position=(3, 0) * U_('µm'),
    dim_fov=(5, 5) * U_('µm')
)

kagome_lattice_fixed = Lattice.generate(boundary=Rect(5.1, 5.1),  u=(.5, 0), v=Vector(.5, 0).rotated(np.pi/3), element=unit_cell, explode=True, remove_outliers=True)

kagome_lattice_fixed_site.create_pattern(
    dim_shape=kagome_lattice_fixed*U_('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

# ----------------------------------------------------------------------------------------------------------------------

honeycomb_lattice_site = sample.create_site(
    dim_position=(-3, 0) * U_('µm'),
    dim_fov=(5, 5) * U_('µm')
)

# def foo(*args):
#     pass

# honeycomb_lattice_fixed = Lattice.generate(boundary=Rect(5.1, 5.1),  u=(.5, 0), v=Vector(.5, 0).rotated(np.pi/3), element=foo)



sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * U_('µm'))
