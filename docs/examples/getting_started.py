# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from fibomat import Sample, Pattern, Mill, U_, Q_
from fibomat import shapes, raster_styles, linalg, default_backends


sample = Sample(description='an optional description for yourself')

site = sample.create_site(
    dim_position=(123, 456) * U_('µm'), dim_fov=(5, 5) * U_('µm'), description='another description'
)

# a mill object with defines the dwell time per spot in the rasterized shape and the number of repeats
single_repeat_mill = Mill(
    dwell_time=Q_('5 ms'), repeats=1
)

# and a line shape
line = shapes.Line(start=(-2, 2), end=(2, 0.5))

# and finally rasterizing style. In this case, the line will be rasterized consecutive from start to end.
line_style = raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)

# everything is collected in a pattern
line_pattern = Pattern(
    dim_shape=line * U_('µm'),
    mill=single_repeat_mill,
    raster_style=line_style
)

# and added to the site.
site += line_pattern
# or
# site.add_pattern(line_pattern)

# secondly, add a square

square = shapes.Rect(width=2, height=2, center=(0, -1))

# rasterize the square line-by-line. see text for details
square_style = raster_styles.two_d.LineByLine(
    line_pitch=10 * U_('nm'),
    scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
    alpha=0,
    invert=False,
    line_style=raster_styles.one_d.Curve(pitch=10 * U_('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
)

# we can also create the pattern in-place
site.create_pattern(
    dim_shape=square * U_('µm'),
    mill=single_repeat_mill,
    raster_style=square_style
)

# plot the patterning layout and save the plot
# if you run the script for yourself, uncomment the following line and delete the line below.
# sample.plot()
sample.plot(fullscreen=_fullscreen)

# export as text file
sample.export(default_backends.SpotListBackend).save('getting_started.txt')
