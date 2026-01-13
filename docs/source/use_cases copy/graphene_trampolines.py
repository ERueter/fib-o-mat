# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat import Vector, Q_, U_, Sample, Mill
from fibomat.units import LengthQuantity, LengthUnit
from fibomat.shapes import ArcSpline, Line, Arc, Circle
from fibomat.layout import Group, Lattice
from fibomat import linalg
from fibomat import raster_styles


def build_trampoline(
    membrane_radius: LengthQuantity,
    trampoline_radius: LengthQuantity,
    bridge_width: LengthQuantity,
    output_unit: LengthUnit
) -> Group:
    membrane_radius = membrane_radius.to(output_unit).m
    trampoline_radius = trampoline_radius.to(output_unit).m
    bridge_width = bridge_width.to(output_unit).m

    trampoline_arc_center = Vector(bridge_width/2 + trampoline_radius, bridge_width/2 + trampoline_radius)

    bridge_length = membrane_radius - bridge_width/2 - trampoline_radius

    if bridge_length < 0.:
        raise ValueError('bridge_length < 0. Choose another set of parameters.')

    # one trampoline segment is build explicitly
    # all others are given by symmetry operations.
    trampoline_segment = ArcSpline.from_segments([
        Line((membrane_radius, bridge_width/2), (membrane_radius - bridge_length, bridge_width/2)),
        Arc.from_points_center(
            start=(membrane_radius - bridge_length, bridge_width/2),
            end=(bridge_width/2, membrane_radius - bridge_length),
            center=trampoline_arc_center,
            sweep_dir=False
        ),
        Line((bridge_width/2, membrane_radius - bridge_length), (bridge_width/2, membrane_radius)),
    ])

    # three different transformation operations can be used to get the other three segments.
    return Group([
        trampoline_segment,
        trampoline_segment.rotated(np.pi/2, origin=(0., 0.)),
        trampoline_segment.scaled(-1, origin=(0., 0.)),
        trampoline_segment.mirrored((1., 0.)).reversed()
    ])


membrane_pattern = Sample()

trampoline_grid = Lattice(nu=3, nv=3, du=5, dv=5)
annotation_grid = Lattice(nu=3, nv=3, du=5, dv=5)

# first: alternate the scale and rotation of trampolines
scales_angles = [1., 2/3, 1/3]
trampoline = build_trampoline(
    membrane_radius=Q_('2 µm'), trampoline_radius=Q_('1.5 µm'), bridge_width=Q_('.1 µm'), output_unit=U_('µm')
)
for i_v in range(3):
    trampoline_grid[0, i_v] = trampoline.transformed(
        linalg.scale(scales_angles[i_v], origin=(0, 0)) | linalg.rotate(np.pi*scales_angles[i_v], origin=(0, 0))
    )

    annotation_grid[0, i_v] = Circle(r=2).scaled(scales_angles[i_v])

trampoline_radii = [1.9, 1, .2]

for i_v in range(3):
    trampoline_grid[1, i_v] = build_trampoline(
        membrane_radius=Q_('2 µm'),
        trampoline_radius=trampoline_radii[i_v] * U_('µm'),
        bridge_width=Q_('.1 µm'),
        output_unit=U_('µm')
    )

    annotation_grid[1, i_v] = Circle(r=2)

bridge_widths = [3/5, 2/5, 1/5]
for i_v in range(3):
    trampoline_grid[2, i_v] = build_trampoline(
        membrane_radius=Q_('2 µm'),
        trampoline_radius=Q_('1 µm'),
        bridge_width=bridge_widths[i_v] * U_('µm'),
        output_unit=U_('µm')
    )
    annotation_grid[2, i_v] = Circle(r=2)


membrane_pattern.add_annotation(annotation_grid)

site = membrane_pattern.create_site(((0, 0), U_('µm')), ((20, 20), U_('µm')))
site.create_pattern(
    (trampoline_grid, U_('µm')),
    Mill(dwell_time=Q_('4 ms'), repeats=1),
    raster_style=raster_styles.one_d.Curve(pitch=Q_('50 pm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
)

membrane_pattern.plot(fullscreen=_fullscreen)
