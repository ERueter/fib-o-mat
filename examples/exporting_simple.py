from fibomat import Sample, U_, Mill, Q_
from fibomat import default_backends, shapes, raster_styles


s = Sample()
site = s.create_site(
    dim_position=(0, 0) * U_('µm'), dim_fov=(5, 5) * U_('µm')
)

mill = Mill(dwell_time=Q_('1 ms'), repeats=4)

site.create_pattern(
    dim_shape=shapes.Line((-2, -2), (2, 2)) * U_('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(
        pitch=Q_('1 nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
    )
)

exported = s.export(default_backends.SpotListBackend)
exported.save('file.txt')

