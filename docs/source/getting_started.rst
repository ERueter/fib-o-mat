Getting started
===============

An installed python version = 3.8 is required.

Create a virtual environment and active it with

.. code-block:: bash

    $ python -m venv .venv
    # for *nix systems
    $ source .venv/bin/activate
    # for MS windows
    $ .venv\Scripts\activate.bat

See `here <https://docs.python.org/3/tutorial/venv.html>`__ for further information on virtual environments.

Install the fib-o-mat package with

.. code-block:: bash

    $ pip install --upgrade fibomat

If this does not work for you, fib-o-mat must be build from source for your system. This is explained
:doc:`here <user_guide/installation>`.


"Hello world" example
---------------------

This getting started guide shows how to create a simple patterning layout.
The layout will consist of a line and rectangular shape.

The complete python file can be found in the git's `example folder <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/getting_started.py>`__.

Starting point of each patterning design is the :class:`~fibomat.sample.Sample` class. This class is the
gluing between everything else in this library. ::

    from fibomat import Sample, Pattern, Mill, U_, Q_
    from fibomat import shapes, raster_styles, linalg, default_backends


    sample = Sample(description='an optional description for yourself')

Next, a patterning site must be defined. The patterning site defines the stage position of the microscope and the field
of view to be used during the patterning process.

(Physical) units, if needed, are specified with the help of the :mod:`fibomat.units` submodule. See the documentation for some examples.
In the patterning site creation process, units are needed to specify the lengths unit for the stage position and the field of view. ::

    site = sample.create_site(
        dim_position=(123, 456) * U_('µm'), dim_fov=(5, 5) * U_('µm'), description='another description'
    )

The site will be located at (123, 456) µm and will have a field of view of (5, 5) µm in horizontal and vertical direction, respectively.
The interpretation of `dim_position` value is dependent on your microscope. For details, consult the documentation of
the used exporting backend (this concept will be introduced a little later).

The library has some predefined shapes which can be used to build patterning geometries.
A list of all shapes is given at :ref:`user_guide/geometric-shapes:geometric shapes`.

The pattering settings are specified with classes given in the :mod:`fibomat.mill` and :mod:`fibomat.raster_styles`
submodules. The former class is used to define the dwell time per spot and the latter class is used to define the rasterization method to be used.
Shapes and patterning settings are collected in class called :class:`~fibomat.pattern.Pattern`. It merges geometries, mill settings and some optional additional parameters.

Putting all together for a line shape ::

   # the mill object defines the dwell time per spot and the total number of repeats
    single_repeat_mill = Mill(
        dwell_time=Q_('5 ms'), repeats=1
    )

    # a single line
    line = shapes.Line(start=(-2, 2), end=(2, 0.5))

    # The used rasterizatio style. the line is rasterized with a constant pitch of 1 nm in a consecutive (linear) way.
    line_style = raster_styles.one_d.Curve(pitch=Q_('1 nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)

    # everything is collected in a pattern
    line_pattern = Pattern(
        dim_shape=line * U_('µm'),
        mill=single_repeat_mill,
        raster_style=line_style
    )


Note that the ``line`` is equipped with a length unit during pattern creation. Otherwise the scaling would not be defined.

.. caution:: The shape position is interpreted always relative to the center of the corresponding site, not to the global coordinate system.

Secondly, a square is added to the sample. In contrast to the line pattern before, the definition of the rasterization style is more complicated.
Here, the square is rasterized line-by-line with lines parallel to the y-axis. The lines are ordered in a consecutive way.
Here, the square is rasterized line-by-line with lines parallel to the y-axis. The lines are ordered in a consecutive way.
Additionally, the rasterization style of the individual lines must be given to. In doing so, a similar style is used as shown in the line pattern shown before.

::

    square = shapes.Rect(width=2, height=2, center=(0, -1))

    # rasterize the square line-by-line. see text for details
    square_style = raster_styles.two_d.LineByLine(
        line_pitch=10 * U_('nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        alpha=0,  # angle of the rasterized lines towards the y axis
        invert=False,  # if True, the rasterization goes from top to bottom and from bottom to top if False
        line_style=raster_styles.one_d.Curve(pitch=10 * U_('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )

    # we can also create the pattern in-place
    site.create_pattern(
        dim_shape=square * U_('µm'),
        mill=single_repeat_mill,
        raster_style=square_style
    )


Finally, the finished pattern layout can be plotted and exported. Exporting is carried out via so called backends.
A backend takes all sites with their contained patterns and creates a file (or something else) a microscope can understand.
Currently, only two backends are  available in the open source library: first, a plotting backend for
visualization and secondly, a rasterization backend. The rasterization backend rasterizes all geometries and creates a
text file with all spots and their corresponding dwell times. ::

    sample.plot()

    # export as text file
    sample.export(default_backends.SpotListBackend).save('foo.txt')

The output format of the SpotListBackend can be customized and is demonstrated :ref:`here <user_guide/exporting_visualization:exporting microscope readable output>`.

The resulting plot is shown below. The blue box shows the field ov view of the patterning site. The yellow shapes
represent the patterning geometries.

Select on one of the zoom tools in the right panel to zoom in and out, either with scrolling or a box selection.
Hovering over the shapes opens a pop-up menu which displays some information about the geometry, mill and rasterization style.

Additionally, use the measure band tool (icon with horizontal arrow) to measure distances and angles.

.. bokeh-plot-link:: ../examples/getting_started.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/getting_started.py

Further, the script exports a file containing the rasterized patterns according to the settings given above. This file ('getting_started.txt') can be visualized with the :ref:`ion beam simulation tool <user_guide/exporting_visualization:ion beam simulation>`.
