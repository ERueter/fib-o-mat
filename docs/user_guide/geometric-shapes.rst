Geometric shapes
================

fib-o-mat supports a decent number of different geometric shapes. This shapes will eventually define the patterning
geometry.

Zero-dim shapes:

    - :class:`~fibomat.shapes.spot.Spot`
    - :class:`~fibomat.shapes.rasterizedpoints.RasterizedPoints`

One/two dim shapes:

    - :class:`~fibomat.shapes.line.Line`
    - :class:`~fibomat.shapes.polyline.Polyline`
    - :class:`~fibomat.shapes.polygon.Polygon`
    - :class:`~fibomat.shapes.arc.Arc`
    - :class:`~fibomat.shapes.arc_spline.ArcSpline`
    - :class:`~fibomat.shapes.parametric_curve.ParametricCurve`
    - :class:`~fibomat.shapes.rect.Rect`
    - :class:`~fibomat.shapes.circle.Circle`
    - :class:`~fibomat.shapes.ellipse.Ellipse`
    - :class:`~fibomat.shapes.hollow_arc_spline.HollowArcSpline`

Meta shapes:
    - :class:`~fibomat.shapes.text.Text`

See the source file of the plot below for usage examples of all shapes.

.. bokeh-plot-link:: ../examples/all_shapes.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/all_shapes.py


Rigid transformations and isotropic scaling
-------------------------------------------

All shapes (except :class:`~fibomat.shapes.parametric_curve.ParametricCurve`) can be translated, rotated, mirrored and scaled.

This transformations can be invoked by calling the :meth:`~fibomat.shapes.shape.Shape.translated`, :meth:`~fibomat.shapes.shape.Shape.rotated`, :meth:`~fibomat.shapes.shape.Shape.mirrored` and :meth:`~fibomat.shapes.shape.Shape.scaled` methods respectively.
Due to the immutability of the objects, the transformation functions return a new object with the transformation applied.

Further, all shapes has a :attr:`~fibomat.shapes.shape.Shape.center` property. This property is the geometric mean of all vertices of an shape if not explicitly stated in the reference.

Below, all transformation functions are shown, exemplary with the line shape.

Translating
+++++++++++
Translating a shape with a certain translation vector: ::

    line = Line(start=(1, 0), end=(2, 1))

    translated_line = line.translated((2, 4.5))
    # translated_line.start == (3, 4.5)
    # translated_line.end == (4, 5.5)

Rotating
++++++++
Rotating a shape around the coordinate origin (0, 0): ::

    line = Line(start=(1, 0), end=(1, 1))

    # rotation around (0, 0) with angle np.pi/2 in mathematical positive direction (counterclockwise)
    rotated_line = line.rotated(np.pi/2)
    # rotated_line.start == (0, 1)
    # rotated_line.end == (-1, 1)

Rotating a shape around a custom origin: ::

    line = Line(start=(1, 0), end=(1, 1))

    # rotation around (3.5, 4.5) with angle np.pi/2 in mathematical positive direction (counterclockwise)
    rotated_line = line.rotated(np.pi/2, origin=(3.5, 4.5))
    # rotated_line.start == (8, 2)
    # rotated_line.end == (7, 2)

Rotating a shape around its current center: ::

    line = Line(start=(1, 0), end=(1, 1))

    # rotation around (3.5, 4.5) with angle np.pi/2 in mathematical positive direction (counterclockwise)
    rotated_line = line.rotated(np.pi/2, origin='center')
    # rotated_line.start == (.5, .5)
    # rotated_line.end == (1.5, .5)

Mirroring
+++++++++

Mirroring along an axis: ::

    line = Line(start=(1, 0), end=(1, 1))

    # mirror the line at y-axis
    mirrored_line = line.mirrored((1, 0))
    # rotated_line.start == (1, 0)
    # rotated_line.end == (1, -1)

Scaling
+++++++
Only isotropic scaling is supported in fib-o-mat (x- and y-axis are always scaled equally).
Scaling supports the ``origin`` parameter similarly to the rotating transformation. ::

    line = Line(start=(1, 0), end=(1, 1))

    scaled_line_1 = line.scaled(2)
    scaled_line_2 = line.scaled(2, origin=(3, 4))  # custom origin
    scaled_line_3 = line.scaled(2, origin='center)  # line's center as origin

Chaining of transformation
++++++++++++++++++++++++++

If multiple transformation can be applied in one step with the :meth:`~fibomat.shapes.shape.Shape.transformed` method.

This method expects chained transformation stubs predefined in the :mod:`~fibomat.linalg` submodule and applies them all in once ::

    # import the stubs
    from fibomat.linalg import translate, rotate, mirror, scale

    line = Line(start=(1, 0), end=(1, 1))

    # translate, rotate, ... expect the same paramters as shown above.
    # these stubs are chained with "|" and are applied from left to right.
    # the 'center' used in the rotate transformation in the center after translating
    transformed_line = line.transformed(
        translate((1, 1)) | rotate(np.pi/2, origin='center') | mirror((1, 0)) | scale(2)
    )

This method is more efficient than calling the transformation methods individually.

Defining a custom pivot position
++++++++++++++++++++++++++++++++

Every shape has a :attr:`~fibomat.shapes.shape.Shape.pivot` property. By default, this property is equal to the :attr:`~fibomat.shapes.shape.Shape.center` property. But contrarily, the pivot can be customized by assigning a function to the property. This function takes the element itself as a parameter ::

    line = Line(start=(1, 0), end=(1, 1))
    # line.pivot == line.center == (1, .5)

    line.pivot = lambda self: self.center + (1, 1)
    # line.pivot == (2, 1.5) != line.center == (1, .5)

    translated_line = line.translated((2, 3))
    # translated_line.pivot == (4, 4.5) != translated_line.center == (3, 3.5)

Additionally, the pivot property can also be used in rotation and scaling transformations as origin ::

    rotated_line = line.rotated(np.pi/2, origin='pivot')
    scaled_line = line.scaled(2, origin='pivot')

The pivot property can be useful especially when dealing with groups and grid layouts (REF).


Parametric Curves
-----------------

:class:`~fibomat.shapes.parametric_curve.ParametricCurve` is a special type of shapes in the library. This shape cannot be transformed.
Even so, it supports a method to approximate a parametric curve to arc spline which, in turn, allows all transformations.

This shape represents parametric functions of type ``f: [a, b] -> R^2``. This function must be element of ``C^3`` (three times continuous differentiable) on the inside of ``[a, b]``. At ``a`` and ``b`` the curve must be only ``C^0``. This means, the curve must not have any kinks, cusps, etc. ``[a, b]`` is the domain of the curve.

The most easiest way of defining a parametric is done with the :meth:`~fibomat.shapes.parametric_curve.ParametricCurve.from_sympy_curve` classmethod.
This class method takes a `sympy <https://sympy.org>`__ curve (:class:`sympy.geometry.curve.Curve`) defining ``f``, all required derivatives and other functions are calculated automatically. ::

    from fibomat.shapes import ParametricCurve

    # sympy example

    import sympy
    from sympy.abc import t
    from sympy import sin, cos

    curve = sympy.Curve(
        [2 * cos(2 * t) + 3 * cos(t), 2 * sin(2 * t) - 3 * sin(t)],
        (t, 0, 2*np.pi)
    )

    parametric_curve = ParametricCurve.from_sympy_curve(curve, try_length_integration=True)
    spline_from_sympy = parametric_curve.to_arc_spline(epsilon=.1)


If ``try_length_integration`` is set to `True`, sympy will try to calculate the arc length function analytically. This will fail for most input functions.

The more complicated method involves defining the function ``f`` and its first two derivatives by hand.

This functions (``f``, ``df/du``, ``d^2f/du^2``) should take a numpy array as parameter (``u``) and must return an numpy array with shape ``(N, 2)`` containing the calculated points where N is the length of u. Optionally, a `curvature <https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization>`__ and a length function can be defined. Even so it is optional, it will speed up thinks significantly.
The curvature function must take an numpy array as parameter (defining ``u``) and return an numpy array with curvatures (same shape as ``u``).
The length function must take to floats as parameters ``u_0`` and ``u_1`` and should return the arc length of ``f`` between ``u_0`` and ``u_1``.

As an example, we take the function ``f(u) = (u, u**2)``. The derivatives are ``f'(u) = (1, 2*u)`` and ``f''(u) = (0, 2)``.
The curvature is given by ``k(u) = 2 / (1 + 4 u**2)`` and the length by ``L(u_0, u_1) = l(u_1) - l(u_0)`` with ``l(u) = u*sqrt(4*u**2 + 1)/2 + asinh(2*u)/4``.

(At some time we will add proper rendering for formulas..).

Defining the required functions can be easily done with numpy. Special care must be taken if any of the derivatives is constant (this case is handled automatically by using the the sympy method) ::

    import numpy as np

    def f(u):
        u = np.asarray(u)

        return np.array(
            (u, u**2)
        ).T


    def df(u):
        u = np.asarray(u)

        return np.array(
            (np.full_like(u, 1), 2*u)
        ).T


    def d2f(u):
        u = np.asarray(u)

        return np.array(
            (np.full_like(u, 1),  np.full_like(u, 2))
        ).T


    def curvature(u):
        u = np.asarray(u)

        return 2 / (1 + 4 * u**2)**1.5


    def length_impl(u):
        return u*np.sqrt(4*u**2 + 1)/2 + np.arcsinh(2*u)/4


    def length(u_0, u_1):
        return length_impl(u_1) - length_impl(u_0)

    parametric_curve = ParametricCurve(
        f, df, d2f,
        domain=(0, 1), bounding_box=None, curvature=curvature, length=length
    )

All parametric curves can be rasterized with equidistant points. The method :meth:`~fibomat.shapes.parametric_curve.ParametricCurve.rasterize` returns the values in the parametric domain where points are located and the :meth:`~fibomat.shapes.parametric_curve.ParametricCurve.rasterize_at` will return the actual points in ``R^2``.

Parametric curves can be approximated by arc splines by calling :meth:`~fibomat.shapes.parametric_curve.ParametricCurve.to_arc_spline`. This method takes as parameter the maximum value the parametric curve and fitted arc spline are allowed to differ.

Both examples can be found at `<https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/parametric_curve.py>`__.

If a parametric curve with cusps should be fitted, the curve can be splitted at the cusp and both halfes should be treated individually.
Finally, these two curves can be stitched together with the :meth:`~fibomat.shapes.arc_spline.ArcSpline.from_segments` class method of the arc spline class ::

    parametric_curve_part_1 = ParametricCurve(...)
    parametric_curve_part_2 = ParametricCurve(...)

    arc_spline = ArcSpline.from_segments([
        parametric_curve_part_1.to_arc_spline(),
        parametric_curve_part_2.to_arc_spline(),
    ])

Operations on :class:`~fibomat.shapes.arc_spline.ArcSpline`\ s
--------------------------------------------------------------

All shapes except spots can be convert to arc splines. For this, every shape has a :meth:`~fibomat.shapes.shape.Shape.to_arc_spline` method.
E.g. ::

    line = shapes.Line(start=(0, 0), end=(1, 1))

    line_as_arc_spline = line.to_arc_spline()

.. warning:: The conversion of parametric curves is still wok in progress an might not work as expected.

All curve tools are implemented in the :mod:`~fibomat.curve_tools` submodule.

Intersections
+++++++++++++

With the :func:`~fibomat.curve_tools.intersections.curve_intersections` function, intersections between two curves can be calculated.
The function returns a directory with keys ``intersections`` and ``coincidences``. The first element contains the
curve intersection and the latter one contains intervals, wehere the two curves are identical.

Additionally, self intersections can be calculated with the :func:`~fibomat.curve_tools.intersections.self_intersections` function

.. bokeh-plot-link:: ../examples/curve_intersections.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/curve_intersections.py


Boolean operations
++++++++++++++++++

fib-o-mat supports 'union', 'xor', 'exclude' and 'intersect' as Boolean operation modes in the
:func:`~fibomat.curve_tools.combine.combine_curves` function. This method returns a dictionary with keys ``remaining`` and ``subtracted``.
Both keys contain as value a list of arc splines.

In the example below, these operations are applied to two overlapping circles.

.. warning:: Boolean operations can only by applied to closed arc splines.

.. bokeh-plot-link:: ../examples/boolean_operations.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/boolean_operations.py

Offsetting
++++++++++

Offsetting can be done with the :func:`~fibomat.curve_tools.offset.inflate` and :func:`~fibomat.curve_tools.offset.deflate` functions.
These funtions will offest a given arc spline outwards and inwards, respectively.
For this, the offset pitch must be defined and the number of offset steps or the total offsetting distance. If a arc
spline is deflated, the number of steps or distance can be left out. In this case, the spline is deflated until no
spline is left.

.. bokeh-plot-link:: ../examples/offsetting.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/offsetting.py


Rasterizing
+++++++++++

If needed, shape outlines can be rasterized manually with the :func:`~fibomat.curve_tools.rasterize.rasterize` function.

For visualization purpose, the rasterized points are displayed with :class:`~fibomat.shapes.spot.Spot`\ s here.

.. bokeh-plot-link:: ../examples/rasterize.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/rasterize.py

Fill with lines
+++++++++++++++

Closed arc splines can be filled with parallel lines with the :func:`~fibomat.curve_tools.rasterize.fill_with_lines` function.

The function sweeps a line over the spline. For every step, the sweeping line is trimmed on the outline of the arc spline. All remaining parts inside the spline are collected in list (this list is called row in the following). The sweeping is repeated until the complete spline is swept.

Besides the enclosing spline and a pitch, this function requires the angle of the filling lines towards the x-axis and a boolean argument called ``invert`` indicating the the sweeping direction. See the figure below for further explanation. The function returns a list of lists where each entry in the outer list contains all lines corresponding to a row.

.. list-table:: Influence of the ``alpha`` and ``invert`` parameters on the filling lines. The big arrows indicate the sweeping direction used to determine the filling lines. The returned rows are sorted ascending in the sweeping direction.

    * - .. figure:: /_static/alpha_ge.png
            :height: 250px

      - .. figure:: /_static/alpha_l.png
            :height: 250px

This methods works also for

.. bokeh-plot-link:: ../examples/fill_with_lines.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/fill_with_lines.py

Smoothing
+++++++++

Non-differentable arc splines can be :func:`~fibomat.curve_tools.smooth`\ ed with arcs to create a smooth spline. Cusps are replaced with arcs of a user defined radius.

.. bokeh-plot-link:: ../examples/smoothing.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/smoothing.py



Text
----

fib-o-mat supports simple text blocks composed of glyphs of the Hershey font family. By default, the glyphs are composed of polylines. Consequently, the text is one dimensional and can be used with any backend supporting polylines.

To create a text, use the :class:`~fibomat.shapes.text.Text` class. USe the `font_size` parameter to scale the text. By default, the height of upper case letters is set to 1.
The text is placed by default at `(0, 0)`. A :class:`~fibomat.shapes.text.Text` is a :class:`~fibomat.layout.groups.Group`. Hence, it supports all type of transformation (e.g. it can be translated).

.. bokeh-plot-link:: ../examples/text.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/text.py


To get further control of the positioning, the :class:`~fibomat.shapes.text.Text` class has the :meth:`~fibomat.shapes.text.Text.baseline_anchor`. With this, the left, right and center positions of the baseline can be accessed.

See the example below for some examples.

.. bokeh-plot-link:: ../examples/text_positioning.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/text_positioning.py


Importing shapes from vector graphic files
------------------------------------------

tbd..
