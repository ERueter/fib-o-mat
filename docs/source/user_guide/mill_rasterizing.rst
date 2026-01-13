Mill & rasterizing settings
===========================

To create a complete :class:`~fibomat.pattern.Pattern`, a :class:`~fibomat.mill.mill.Mill` and
:class:`~fibomat.raster_styles.rasterstyle.RasterStyle` must be defined along with a patterning shape.

In a pattern, the following pieces of information are collected:

    1. what should be rasterized (geometric shape)
    2. in which way the shape should be rasterized (rasterization style, pitches)
    3. how the rasterized shape should be milled (dwell time and current)

Defining a mill
---------------
In the most simple case, the :class:`~fibomat.mill.mill.Mill` takes the patterning current and the number of total repeats of a shape as attributes. ::

    from fibomat import Mill, Q_

    mill = Mill(current=Q_('1 pA'), repeats=5)

|:test_tube:| Providing custom parameters to a mill object
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This can be useful in combination with a custom exporting or custom rasterization styles.

In the extending section, an example is given.


..    The mill object can also store custom settings. To add them to a Mill object, use the :meth:`~fibomat.mill.Mill.special_mill` classmethod.
    In addition to the current and number of repeats, arbitrary other parameters can be passed to the Mill object. These are stored in the class an can be accessed in custom patterning backend (REF) for example. ::

..        special_mill = Mill.special_mill(current=Q_('1 pA'), repeats=5, use_flood_gun=True, defocus=20)

..        # ...

..        # access the extra parameters at a later point
..        print(special_mill.use_flood_gun)
        print(special_mill.defocus)

..    The example above illustrates how extra parameters can be passed and retrieved at a later stage (in this case the parameters 'use_flood_gun' and 'defocus').

..    .. note:: All current implemented backends in fib-o-mat ignore all extra parameters.

..    |:test_tube:| Defining an ion beam shape
  ++++++++++++++++++++++++++++++++++++++++

    For dose calculation or the :func:`~fibomat.optimize.optimize` routine for example, the beam shape must be known.
    For this, the fib-o-mat package provides the :class:`~fibomat.mill.ionbeam.GaussBeam` class. As the name indicates, this class describes the ion beam with a Gaussian shape. In the extending fib-o-mat section (REF) it is explained, how a custom beam profile is defined.

    A GaussianBeam is defined by the full-width-half-maximum beam width and the total beam current. ::

        from fibomat import mill

        beam = mill.GaussBeam(fwhm=Q_('3 nm'), current=Q_('1 pA'))

    The GaussianBeam class provides some utility methods which are explained in the code snipped below ::

        # returns the standard deviation of the distribution
        print(beam.std)

        # Calculate the ion flux at position (0, 0) µm with spots at (-1, 0), (0, 0), (1, 0) µm
        # hence the influence of surrounding spots of the spot at (0, 0) is calculated.
        print(beam.flux_at(
            (0, 0),
            [(-1, 0), (0, 0), (1, 0)],
            U_('µm')
        ))

        # Calculate the ion flux of a single, isolated spot.
        # this is the same as calling beam.flux_at((0, 0), (0, 0), U_('µm'))
        print(beam.nominal_flux_per_spot())

        # Calculate the ion flux of a spot on line with pitch 1 nm
        print(beam.nominal_flux_per_spot_on_line(Q_('1nm))

        # Calculate the ion flux of a spot on rectangle with pitches 1 nm, 1 nm in x and y directions, respectively.
        print(beam.nominal_flux_per_spot_in_rect(Q_('1nm), Q_('1nm))


    The ``nominal_flux_*`` methods can be used to calculate a nominal flux to be used in the optimization routine (see below here REF) or to calculate the ion dose on the regular rasterized line/grid.


Specifying the rasterization style
----------------------------------

The rasterization styles define, how a shape should be rasterized. For different dimensions, different raster styles are pre-defined. The creation of custom patterning style is explained elsewhere REF.

The default rasterization styles in the fib-o-mat package are introduced in the following.

The subsection refer to examples in the git repository at `<https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles>`__. These can be executed by

.. code-block:: bash

    $ python examples/raster_styles/spot.py && beam_simulation rasterized.txt

if the current directory is the root of the fib-o-mat repository. ``spot.py`` can be replaced by all other scripts in the ``examples/raster_styles`` directory. See also :ref:`ion beam simulation <user_guide/exporting_visualization:ion beam simulation>`.

Zero-dim
++++++++

    - :class:`~fibomat.raster_styles.zero_d.singlespot.SingleSpot`
    - :class:`~fibomat.raster_styles.zero_d.prerasterized.PreRasterized`

Both zero-dimensional raster style do not take any parameters. These raster styles can only be used for :class:`~fibomat.shapes.spot.Spot`\ s and pre-rasterized objects (:class:`~fibomat.shapes.rasterizedpoints.RasterizedPoints` and :class:`~fibomat.rasterizedpattern.RasterizedPattern`), respectively.

Examples:
    * `single spots <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/spot.py>`__
    * |:test_tube:| `manual rasterization <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/pre_rasterized.py>`__

One-dim
+++++++

The only raster style for 1-dim shapes is the :class:`~fibomat.raster_styles.one_d.curve.Curve` style.
This style expects a pitch (distance between neighboring spots) and scan style. All three possible scan styles are visualized below.

.. list-table:: Available scan styles for 1-dim shapes.

    * - .. figure:: /_static/consecutive_1d.png
            :height: 250px

      - .. figure:: /_static/back_stitch_1d.png
            :height: 250px

      - .. figure:: /_static/back_and_forth_1d.png
            :height: 250px

Examples:
    * `all 1-dim styles <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/one_dim.py>`__

Two-dim
+++++++

fib-o-mat includes two different rasterizing methods of two-dim shapes (:class:`~fibomat.raster_styles.two_d.linebyline.LineByLine` and :class:`~fibomat.raster_styles.two_d.contour_parallel.ContourParallel`).
Both rasterization styles fill a given shape with lines or curves. The ordering of these lines and curves is defined by a scan style.
All available scan styles are shown below.


:class:`~fibomat.raster_styles.two_d.linebyline.LineByLine` rasterization
*************************************************************************

The line-by-line rasterizing style rasterizes a closed shape by sweeping a line over it. This style is commonly supported in other (proprietary) patterning software.

Details on the method can found at the description of the fill_with_lines method :ref:`here <fill with lines>`.

.. list-table:: Available scan styles for 2-dim shapes.

    * - .. figure:: /_static/consecutive_2d.png
            :height: 250px

      - .. figure:: /_static/cross_section_2d.png
            :height: 250px

    * - .. figure:: /_static/serpentine_2d.png
            :height: 250px

      - .. figure:: /_static/double_serpentine_2d.png
            :height: 250px

    * - .. figure:: /_static/double_serpentine_same_path_2d.png
            :height: 250px

      - .. figure:: /_static/back_stitch_2d.png
            :height: 250px


The scan sequences in the figure above only define the ordering of the individual 1-dim shapes which fill the 2-dim shape.
In the plot above, the 2-dim shape is a rectangle filled by 1-dim lines.
Hence, the 2-dim rasterization styles require also a 1-dim rasterization style as parameter (among others) which will be used for the filling shapes.


:class:`~fibomat.raster_styles.two_d.contour_parallel.ContourParallel` offset rasterization
********************************************************************************************

This style generate contour-parallel offsetted curves of the passed shape to rasterized it.

.. |:test_tube:| To decrease the influence of artifacts due to offsetting, this rasterizing styles supports optimizing of the rasterized dwell points. See the use case :ref:`Plasmonic tetramer antennas based on single-crystalline gold flakes` for an usage example of the optimization process.


Examples:
    * `various LineByLine styles <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/line_by_line.py>`__
    * `various ContourParallel styles <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/contour_parallel.py>`__

.. * ContourParallel with optimizations: `<https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/contour_parallel_optimizations.py>`__

