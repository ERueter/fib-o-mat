import numpy as np

from fibomat import shapes
from fibomat import sample

sp = sample.Sample('All shapes')

# spot
sp.add_annotation(shapes.Spot((.5, .5)))

# line
sp.add_annotation(shapes.Line((-.25, .25), (.25, .75)).translate((1.5, 0)))

# polyline
poly_points = []
for t in np.arange(1, 5, 0.25):
    poly_points.append((.5 * np.cos(3 * t) / t, .5 * np.sin(3 * t ) / t))
sp.add_annotation(shapes.Polyline(poly_points).translate((2.5, .5)))

# arc
sp.add_annotation(shapes.Arc.from_points((-.25, .25), (.0, .6), (.25, .75)).translate((.5, -1)))

# curve
curve = shapes.Curve([
    shapes.Arc.from_points((-.25, .25), (.0, .8), (.25, .75)),
    shapes.Line((.25, .75), (0, .4))
]).translate((1.5, -1))
sp.add_annotation(curve)

# polygon
sp.add_annotation(shapes.Polygon.regular_ngon(6, .75/2).translate((2.5, -.5)), filled=True)

# rect
sp.add_annotation(shapes.Rect(.6, .6, 0.3, (.5, -1.5)), filled=True)
# circle
sp.add_annotation(shapes.Circle(.3, (1.5, -1.5)), filled=True)

# ellipse
sp.add_annotation(shapes.Ellipse(.4, .2, -0.3, (2.5, -1.5)), filled=True)

sp.plot(show=True, hide_sites=True, fullscreen=False)
