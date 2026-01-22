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
The milling behaviour for a pattern is specified by a ``Mill`` object. The current implementation provides several related classes in
``fibomat.mill.mill``:

- ``Mill`` — simple, constant dwell time per spot and number of repeats
- ``DDDMill`` — base for a dwell-time-per-point function and repeats (used for position-dependent dwell)
- ``MatrixMill`` — use a 2D dwell-time matrix (image) as a rasterized dwell map
- ``SpecialMill`` — lightweight container for custom/back-end specific parameters
-  ``SILMill`` — example customized mill for fabrication of solid immersion lenses

Simple constant dwell time
~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a fixed dwell time per spot, create a ``Mill`` with a quantity and an integer number of repeats::

    from fibomat import Mill, Q_

    mill = Mill(dwell_time=Q_('1 microsecond'), repeats=1)

Note that ``Mill`` stores the dwell-time as a (constant) function internally (it subclasses ``DDDMill``). If only milling one pattern, normally ``repeats`` will be 1 and multiple repeats of the milling process will instead be marked in the exported file in the end.

Position-dependent dwell (DDDMill)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the dwell time depends on the physical location, use a ``DDDMill``-style object which holds a function
``dwell_time(point: np.ndarray) -> Quantity`` and an integer ``repeats``. The provided ``Mill`` above is a convenience
wrapper that builds a constant function for you. You can also subclass or construct a ``DDDMill`` directly.

Example of a custom DDD mill (callable returns a pint quantity)::

    def dwell_func(point):
        x, y = point[0], point[1]
        return Q_(max(0.1, 1.0 - (x**2 + y**2)), 'microsecond')

    from fibomat.mill.mill import DDDMill
    custom = DDDMill(dwell_time=dwell_func, repeats=1)


Matrix-based dwell map: ``MatrixMill``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``MatrixMill`` allows supplying a 2D numpy array of dwell values (e.g. from an image) together with a pixel size ``dx``
and an ``origin``. It converts a point (x, y) to matrix indices and returns the corresponding dwell as a pint quantity.

.. warning:: ``MatrixMill`` is still under development and not yet fully supported.

Extra/back-end parameters (SpecialMill)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you need to bundle arbitrary extra parameters for a back-end, use ``SpecialMill`` which is a small container built on
``MillBase``. Customized ack-ends may choose to read these attributes but note that built-in back-ends probably won't work with this mill.

Use-case specific: ``SILMill``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``SILMill`` is a specialised DDDMill that implements a radial dwell profile for a SIL geometry. Find documentation on usage in :doc:`../use_cases/sil_mill_guide` .

Defining an ion beam shape
--------------------------
For dose calculations or optimization routines the beam shape is required. The package provides ion beam descriptions
under ``fibomat.mill.ionbeam`` (for example ``GaussBeam``). A Gaussian beam is constructed from a full-width-half-maximum
and a current::

    from fibomat import Q_
    from fibomat.mill import ionbeam

    beam = ionbeam.GaussBeam(fwhm=Q_('3 nm'), current=Q_('1 pA'))

The beam classes provide helper methods such as ``std``, ``flux_at`` and the various ``nominal_flux_*`` convenience methods
used for dose/optimization estimates.


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

:class:`~fibomat.raster_styles.two_d.spiral.Spiral` spiral rasterization
**************************************************************************
The spiral rasterstyle rasterizes a shape by taking the circumcircle of the shape's bounding box, then filling this circle with an archimedian spiral and rasterizing this spiral as a 1-dim shape. This style expects two pitch-parameters: The pitch determines with what pitch the spiral should be rasterized, the spiral pitch determines how dense the spiral arms are supposed to be. Furthermore the user can decide in which direction the spiral shall be rasterized: From the inside to the outside, in reverse direction, or from inside to outside and then back in the reversed sense. See :doc:`../use_cases/sil_mill_guide` for an example. 
.. note:: The spiral rasterstyle is still under development and so far, no matter which scan style the user passes over, the consecutive scan is used. Also there might be problems with complex shapes, eg. shapes with holes. Please report errors via git issue.





Examples:
    * `various LineByLine styles <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/line_by_line.py>`__
    * `various ContourParallel styles <https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/contour_parallel.py>`__

.. * ContourParallel with optimizations: `<https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/raster_styles/contour_parallel_optimizations.py>`__

