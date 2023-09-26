Magnetic Patterning of Co/Pt Multilayer Films
=============================================

This example demonstrates how large patterns with a lot of individual shapes can be easily generated with the fib-o-mat
package. The generated pattern was used to create local modifications in magnetic multilayer films.
In doing so, a variety of different shapes and doses must be covered.
This is achieved by extensive use of the :class:`~fibomat.layout.grid.Lattice` feature of fib-o-mat.

All patterning sites are arranged in grids as well as some of the patterns itself are build up by grids.

The pattern was exported with a custom backend which is (up to now due to license uncertainties) not part of the
fib-o-mat package. The backend exports all patterns to a format which can be directly as a 'step-and-repeat list' in the
NPVE patterning software shipped with Zeiss Orion Nanofab. Each site in the generated pattern is patterned separately.
All stage moves and adjustemnts of the field of view are handled by the NPVE software.
This allows user-independent patterning of large patterns. In
this case, the patterning process took several hours.

The full python file to generate the pattern can be found at gitlab here (REF). The plot below shows the complete
pattern.

.. .. bokeh-plot-link:: ../use_cases/copt_multilayer.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/use_cases/copt_multilayer.py
