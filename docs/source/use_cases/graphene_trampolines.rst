Mechanical resonators based on suspended single-layer graphene
==============================================================

In this use case it is demonstrated, how fib-omat can be used to generate patterns based one a few user set constrains.
This allows to quickly generate pattern which follow certain pre-defined rules without recreating the shape manually.
Instead, the shape is calculated according to user defined rules and constrains which itself may only depend on a few
parameters.

In the present example, trampoline shapes should be cut out of graphene which sits on a substrate with circular holes.
The pattern geometry is depicted in the figure on the right. The grey shaded area is the graphene trampoline which
should remain after patterning. Everything should be cut away. This is achieved by creating cuts on the four trampoline
arms plotted in orange. This causes the four non-trampoline areas to fold in a way that these are out of the way for
the wanted measurements.

.. figure:: /_static/trampoline.png
    :align: center
    :width: 250px

    Pattern geometry for graphene trampolines. The grey shaded area is the trampoline which should remain after
    patterning. The orange lines depict the cuttings made to achieve this. The only required paramters to generate the
    patterns are the trampoline radius and the bridge width.

The geometry per trampoline arm consist of two line segment and a circular arc (cf. the figure above). The
exact positions of these is specified only by the trampoline radius und the bridge width. By defining these the complete
pattern is generated. Additionally, the substrate hole radius must be set which was constant in this case.

The plot below shows different variations of the trampoline patterning changing the scale, rotation, bridge width and
trampoline radius.

.. .. bokeh-plot-link:: ../use_cases/graphene_trampolines.py
    :url: https://gitlab.com/viggge/fib-o-mat/-/blob/master/use_cases/graphene_trampolines.py
