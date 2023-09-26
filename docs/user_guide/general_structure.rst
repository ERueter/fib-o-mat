General structure
=================

The base and root o all other objects in th fib-o-mat package is the :class:`~fibomat.sample.Sample` class.
All other objects are added directly or indirectly to an instance of this class.

To a sample, :class:`~fibomat.site.Site` objects can be added. Sites have a
position and field of view as properties and :class:`~fibomat.pattern.Pattern`\ s can be added to a site.

A pattern is a collection of a patterning geometries, beam settings and rasterizing settings.

A sample can have a unlimited number of sites. The same holds for sites and patterns.

The following example can used as a template for any project. ::

    from fibomat import Sample, Vector, U_

    sample = Sample()

    # add a site with position (5 µm, 6 µm) and a field of view of (2 µm, 2 µm)
    first_site = sample.create_site(
        dim_position = (Vector(5, 6), U_('µm'),
        dim_fov = (Vector(2, 2), U_('µm')
    )

    # see the following chapters to see how shapes, mills, and rasterizing_settings can be defined.
    first_site.create_pattern(
        dim_shape=...,
        mill=...,
        rasterizing_style=...
    )

    # plot the pattern design
    sample.plot()

    # export the pattern. See the corresponding chapter for more details.
    exported = sample.export(...)
    exported.save('my_file.txt')

Sites
-----

A :class:`~fibomat.site.Site` contains the stage position and field-of-view (fov) to be used during patterning as well as all patterns.

The stage position and fov must be passed during creation of a Site object. Though, the fov is optional and is calculated automatically based on the bounding boxes of all added patterns.

.. warning:: Currently, no checks are performed if all passed patterns fit inside a defined fov!

The placement of the patterns is relative to the center of site, not the global coordinate system!
For example: Suppose a site with center (-1, -1) is created and a pattern containing a single spot with position (0, 0) is added. The absolute position of the spot is (-1, -1) with regard to the global coordinate system.


Patterns
--------

:class:`~fibomat.pattern.Pattern` collect geometric :class:`~fibomat.shapes.shape.Shape`\ s, :class:`~fibomat.mill.mill.Mill` settings and :class:`~fibomat.raster_styles.rasterstyle.RasterStyle`\ s.

In doing so, the passed shape must be equipped with a length unit. This can be done by multipling a shape with a unit. ::

    from fibomat import Pattern, U_, Mill
    from fibomat import shapes, raster_styles

    # ...

    pattern = Pattern(
        dim_shape=shapes.Rect(width=1, height=1) * U_('µm'),
        mill=Mill(dwell_time=1 * U_('µs'), repeats=3),
        raster_style=raster_styles.one_d.Curve(
            pitch=1 * U_('µm'), scan_style=raster_styles.ScanStyle.CONSECUTIVE
        )
    )

    site += pattern
    # or use in-place creation of pattern with site.create_patter(dim_shape=..., mill=..., raster_style=...)

|:test_tube:| Create patterns only for plotting or svg export
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For these tasks, the mill can be set to ``None`` and the raster style to a special class defined in :class:`~fibomat.default_backends`. ::

    from fibomat import Pattern, U_, Mill
    from fibomat import shapes, raster_styles, default_backends

    # ...

    pattern = Pattern(
        dim_shape=shapes.Rect(width=1, height=1) * U_('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(dim=1)  # dim=1 => only boundary is plotted. dim=2 => shape is filled.
    )

    site += pattern
    # or use in-place creation of pattern with site.create_patter(dim_shape=..., mill=..., raster_style=...)


.. note:: To add annotations to a plot (i.e. physical dimensions of a sample) which are not part of a pattern but usefull to display, use the :meth:`~fibomat.sample.Sample.add_annotation` method within the Sample class.


Transformation of patterns and sites
------------------------------------
Patterns and sites support the same transformations as introduced in :ref:`user_guide/geometric-shapes:Rigid transformations and isotropic scaling`.

In difference to shapes, sites and patterns are dimensioned.
Therefore, :class:`~fibomat.linalg.vectors.dim_vector.DimVector` must be used instead of :class:`~fibomat.linalg.vectors.vector.Vector` for any transformation method. To illustrate this ::
   from fibomat import Pattern, Vector, DimVector

   pattern = Pattern(...)
   rotated_pattern = pattern.rotated(np.pi/3, center=DimVector(1 * U_('µm'), -2 * U_('µm')))

Further, sites can only be rotated by multiples of 90° and mirrored on the x and y axis as well as their bisectors.

.. warning:: If sites are transformed, all contained patterns will be transformed as well. Newly added patterns after transformation will not be transformed.
