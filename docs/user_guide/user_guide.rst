User Guide
==========

The User Guide introduces all features of the fib-o-mat package.

In the following, the basic concepts in th fib-o-mat library are summarized. Herein, the focus is set to the programming
part. To get more background information, see the associated publication linked at the :ref:`starting page <index:fib-o-mat>`.

In short, fib-o-mat is python library to generate patterns for ion beam instruments. This is implemented as a two step
process. First, a pattering shape must be defined. In a second step, the shape is equipped with beam and rasterizing
settings.
The final pattern design can be exported to a microscope readable format.

During pattern creation, two different paths can be taken. The first one is called 'high level approach'. This means,
that the shape and rasterizing settings are defined and fib-o-mat will do the rasterizing process automatically.

Alternatively, the shapes can be rasterized by hand. This allows the user to have very fine controlled on pattern
designs and apply optimizations to the rasterized points for example ('low level approach').

Of course, both approaches can be combined, too.


.. Additionally, fibomat allows to generate microscope specific files containing all necessary patterning information.
   This patterning information can be rasterized point with dwell times but also geometric definitions of patterning shapes and rasterization styles. If the latter file format type is used, the actual rasterizing process is done by the microscope software and not within fibomat. In this case, only the 'high level approach' can be used.

All sections marked with a test tube |:test_tube:| describe advanced features of the package which may be skipped on first usage.


.. warning:: Currently, fibomat does not contain specific exporting backend to generate microscope readable files. This must
             be added by the user and is explained :ref:`somewhere else <user_guide/extending:extending fib-o-mat>`.
             Even so, the provided backends can be easily modified to support common microscopes.

.. warning:: Some parts of the user guide are still missing and will be added soon.

.. toctree::
    :maxdepth: 2
    :glob:
    :hidden:

    Installation <installation>
    Preliminary <preliminary>
    General structure <general_structure>
    Geometric shapes <geometric-shapes>
    Mill & rasterizing settings <mill_rasterizing>
    Exporting & visualization <exporting_visualization>
    Grouping & arranging <grouping_arranging>
    Extending fib-o-mat <extending>
    Development <development>

