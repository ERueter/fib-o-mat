import numpy as np

from fibomat import Sample, U_, Q_, Mill
from fibomat import default_backends, shapes, curve_tools, raster_styles

sample = Sample()

mill = Mill(dwell_time=Q_('1 ms'), repeats=5)

site_consecutive = sample.create_site(
    dim_position=(0, 0) * U_('µm'),
    dim_fov=(1, 1) *U_('µm')
)

site_consecutive.create_pattern(
    dim_shape=shapes.Line(start=(-.75, -.75), end=(.75, .75)) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
)

# ----------------------------------------------------------------------------------------------------------------------

site_back_stitch = sample.create_site(
    dim_position=(1, 0) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_back_stitch.create_pattern(
    dim_shape=shapes.Circle(r=.5) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(pitch=Q_('100 nm'), scan_sequence=raster_styles.ScanSequence.BACKSTITCH)
)

# ----------------------------------------------------------------------------------------------------------------------

site_back_and_forth = sample.create_site(
    dim_position=(2, 0) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_back_and_forth.create_pattern(
    dim_shape=shapes.Line(start=(-.75, -.75), end=(.75, .75)) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.BACK_AND_FORTH)
)


sample.export(default_backends.SpotListBackend).save('rasterized.txt')

