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
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=Q_('50 nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=Q_('50 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_cross_section = sample.create_site(
    dim_position=(1, 0) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_cross_section.create_pattern(
    dim_shape=shapes.ArcSpline([(-.4, 0, 0), (.4, 0, 1)], is_closed=True) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=Q_('50 nm'),
        scan_sequence=raster_styles.ScanSequence.CROSSECTION,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=Q_('50 nm'), scan_sequence=raster_styles.ScanSequence.BACKSTITCH)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_serpentine = sample.create_site(
    dim_position=(0, -1) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_serpentine.create_pattern(
    dim_shape=shapes.Rect(width=0.5, height=.25) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=Q_('50 nm'),
        scan_sequence=raster_styles.ScanSequence.SERPENTINE,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=Q_('50 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_double_serpentine = sample.create_site(
    dim_position=(1, -1) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_double_serpentine.create_pattern(
    dim_shape=shapes.Polygon.regular_ngon(radius=.4, n=6, center=(0, 0)) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=Q_('50 nm'),
        scan_sequence=raster_styles.ScanSequence.DOUBLE_SERPENTINE,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=Q_('50 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_cross_section = sample.create_site(
    dim_position=(0, -2) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_cross_section.create_pattern(
    dim_shape=shapes.Circle(r=.4) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=Q_('50 nm'),
        scan_sequence=raster_styles.ScanSequence.CROSSECTION,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=Q_('50 nm'), scan_sequence=raster_styles.ScanSequence.BACK_AND_FORTH)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_double_serpentine_same_path = sample.create_site(
    dim_position=(1, -2) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

site_double_serpentine_same_path.create_pattern(
    dim_shape=shapes.Rect(width=.7, height=.35) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=Q_('50 nm'),
        scan_sequence=raster_styles.ScanSequence.DOUBLE_SERPENTINE_SAME_PATH,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=Q_('50 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)


sample.export(default_backends.SpotListBackend).save('rasterized.txt')

