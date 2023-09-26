import numpy as np

from fibomat import Sample
# some other imports which will be used later
from fibomat import units
from fibomat import mill
from fibomat import shapes
from fibomat import pattern

dose_test_sample = Sample('an optional description for yourself')

site = dose_test_sample.create_site(
    position=(123, 456), position_unit=units.U_('µm'), fov=(5, 5), description='another optional description'
)

line = shapes.Line(start=(-2, 2), end=(-1, 2))
line_mill = mill.LineMill(
    dwell_time=units.Q_('5 ms'), current=units.Q_('1 pA'), intra_path_spacing=units.Q_('0.5 nm'), repeats=1
)

line_pattern = pattern.Pattern(shape=line, shape_unit=units.U_('µm'), mill=line_mill)
site += line_pattern
# or
# site.add_pattern(line_pattern)

line.translate((0, -1))

for i, angle in enumerate(np.linspace(0, np.pi/2, 4, endpoint=False)):
    # rotate the line around its center point and translate it
    rotated_line = line.clone().rotate(angle, 'center').translate((i, 0))

    site += pattern.Pattern(shape=rotated_line, shape_unit=units.U_('µm'), mill=line_mill)


spot_mill = mill.SpotMill(dwell_time=units.Q_('5 ms'), current=units.Q_('1 pA'), repeats=1)

for x in np.linspace(-1, 1, 10):
    for y in np.linspace(-2, 0, 10):
        spot = shapes.Spot(position=(x, y))
        site += pattern.Pattern(shape=spot, shape_unit=units.U_('µm'), mill=spot_mill)

# plot the patterning layout and save the plot
dose_test_sample.plot(show=True, filename='foo.html', fullscreen=False)

# export as text file
# not working currently
# dose_test_sample.export('...').save('spot_list.txt')

