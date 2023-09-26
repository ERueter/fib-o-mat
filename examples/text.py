from fibomat import Sample, U_
from fibomat.shapes import Text

hello = Text('Hello fib-o-mat!')

increased_font_size = Text('My font size is 2 units.', font_size=2)
increased_font_size = increased_font_size.translated((0, 2))

s = Sample()

s.add_annotation(hello * U_('µm'))

s.add_annotation(increased_font_size * U_('µm'))


s.plot()

