import numpy as np

from fibomat import Sample, U_, Q_, Mill
from fibomat import default_backends, shapes, curve_tools, raster_styles

sample = Sample()

mill = Mill(dwell_time=Q_('1 ms'), repeats=5)

site_consecutive = sample.create_site(
    dim_position=(0, 0) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_consecutive.create_pattern(
    dim_shape=shapes.Circle(r=.4) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.ContourParallel(
        offset_pitch=Q_('10 nm'),
        offset_direction='outwards',
        offset_distance=Q_('60 nm'),
        start_direction='inwards',
        scan_sequence=raster_styles.ScanSequence.DOUBLE_SERPENTINE,
        line_style=raster_styles.one_d.Curve(pitch=Q_('50 nm'), scan_sequence=raster_styles.ScanSequence.BACKSTITCH),
        include_original_curve=True,
        optimize=False
    )
)

sample.export(default_backends.SpotListBackend).save('rasterized.txt')
