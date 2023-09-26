import numpy as np

from fibomat import Sample, U_, Q_, Mill
from fibomat import default_backends, shapes, curve_tools, raster_styles

sample = Sample()

site = sample.create_site(
    dim_position=(0, 0) * U_('µm'),
    dim_fov=(1, 1) * U_('µm')
)

# create 100 randomly distributed spots and add them to the sample
# each spot is repeated 5 times (which is actually the same as setting the repeats=1, dwell_time=5 ms)
mill = Mill(dwell_time=Q_('1 ms'), repeats=5)

for pos in np.random.uniform(-1, 1, 100).reshape(-1, 2):
    site.create_pattern(
        dim_shape=shapes.Spot(pos) * U_('µm'),
        mill=mill,
        raster_style=raster_styles.zero_d.SingleSpot()
    )

sample.export(default_backends.SpotListBackend).save('rasterized.txt')

