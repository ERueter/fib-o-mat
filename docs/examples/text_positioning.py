from fibomat import Sample, U_
from fibomat.shapes import Text

centered_at_origin = Text('centered at origin')
centered_at_origin = centered_at_origin.translated(-centered_at_origin.center)

baseline_centered_at_y_axis = Text('baseline_centered_at_y_axis')
baseline_centered_at_y_axis = baseline_centered_at_y_axis.translated(
    -baseline_centered_at_y_axis.baseline_anchor('center')
).translated(
    (0, 2)
)

baseline_left_at_y_axis = Text('baseline_left_at_y_axis')
baseline_left_at_y_axis = baseline_left_at_y_axis.translated(
    -baseline_left_at_y_axis.baseline_anchor('left')
).translated(
    (0, 4)
)

baseline_right_at_y_axis = Text('baseline_right_at_y_axis')
baseline_right_at_y_axis = baseline_right_at_y_axis.translated(
    -baseline_right_at_y_axis.baseline_anchor('right')
).translated(
    (0, 6)
)


s = Sample()

s.add_annotation(centered_at_origin * U_('µm'))
s.add_annotation(baseline_centered_at_y_axis * U_('µm'))
s.add_annotation(baseline_left_at_y_axis * U_('µm'))
s.add_annotation(baseline_right_at_y_axis * U_('µm'))



s.plot()

