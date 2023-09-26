# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from fibomat import Sample, Vector, Site, U_, Q_, Mill
from fibomat import shapes, layout, raster_styles

# from fibomat.default_backends.bokeh_backend import StubAreaStyle

import numpy as np

copt_multilayer = Sample('Co/Pt multilayer, 250 µm membrane')

copt_multilayer.add_annotation(shapes.Rect(width=250, height=250, theta=0), description='membrane')

nu = 5
nv = 2 + 3 + 3
membrane_sites_rects = layout.DimLattice(nu=nu, nv=nv, du=Q_('30 µm'), dv=Q_('30 µm'))

# First: just big rectangles with different doses
doses = np.linspace(50, 50 + 50*(nu-1), nu)
for i_u in range(nu):
    for i_v in range(2):
        site = Site(([0, 0], U_('µm')), ([20, 20], U_('µm')))
        site.create_pattern(
            (shapes.Rect(10, 10, 0, center=(0, 0)), U_('µm')),
            Mill.special_settings(dwell_time=Q_('-1 ms'), repeats=-1, dose=doses[i_u] * U_('ions / nm**2')),
            raster_styles.two_d.Linear(pitch_u=Q_('1 nm'), pitch_v=Q_('1 nm'))
        )
        membrane_sites_rects[i_u, i_v] = site

# Second: vertical lines with different width and spacings
widths = [0.3, 0.5, 0.7]

offset = 2
for i_u in range(nu):
    for i_v in range(3):
        site = Site(([0, 0], U_('µm')), ([20, 20], U_('µm')))

        # number of lines = int(fov / (2*widths), fov = 20
        line_grid = layout.Lattice(
            nu=1, nv=int(20 / (2*widths[i_v])),
            du=1, dv=2*widths[i_v],
            element=shapes.Rect(width=widths[i_v], height=20, theta=0)
        )

        site.create_pattern(
            (line_grid, U_('µm')),
            Mill.special_settings(dwell_time=Q_('-1 ms'), repeats=-1, dose=doses[i_u] * U_('ions / nm**2')),
            raster_styles.two_d.Linear(pitch_u=Q_('1 nm'), pitch_v=Q_('1 nm'))
        )
        membrane_sites_rects[i_u, i_v+offset] = site

# third: circles with different diameters in grid array
diameters = [0.2, 0.35, 0.5]

offset = 5
for i_u in range(nu):
    for i_v in range(3):
        site = Site(([0, 0], U_('µm')), ([20, 20], U_('µm')))

        # number of lines = int(fov / (2*widths), fov = 10
        n_circles = int(10 / (2*diameters[i_v]))
        line_grid = layout.Lattice(
            nu=n_circles, nv=n_circles,
            du=2*diameters[i_v], dv=2*diameters[i_v],
            element=shapes.Circle(r=diameters[i_v]/2)
        )

        site.create_pattern(
            (line_grid, U_('µm')),
            Mill.special_settings(dwell_time=Q_('-1 ms'), repeats=-1, dose=doses[i_u] * U_('ions / nm**2')),
            raster_styles.two_d.Linear(pitch_u=Q_('1 nm'), pitch_v=Q_('1 nm'))
        )
        membrane_sites_rects[i_u, i_v+offset] = site

copt_multilayer.add_site(membrane_sites_rects)

copt_multilayer.plot(fullscreen=_fullscreen, legend=_fullscreen)
