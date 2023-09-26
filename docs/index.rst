=====================================================================
fib-o-mat
=====================================================================

A python toolbox to generate focused ion beam patterning layouts
----------------------------------------------------------------

.. figure:: /_static/flowchart.png
    :align: center

fib-o-mat is a Python library to create beam patterns for focused ion beam instruments.

Features:
    * build-in modeling of patterning geometries
    * customizable rasterization styles
    * optimization of patterning geometries and rasterized patterns
    * extendable

Pattern geometries can be modeled directly in Python based on (pre-)defined geometric primitives or imported from vector graphics. These can be equipped with beam and rasterizing settings and exported to microscope compatible files.

fib-o-mat is by designed flexible and easily expandable. Hence, adding support for different microscopes, custom geometric primitives or optimization routines is a straightforward process.

For the usage of fib-o-mat, basic python knowledge and good understanding of the target microscope are mandatory.
See :ref:`getting_started:getting started` for an introduction to this library and the :ref:`user_guide/user_guide:user guide` for a complete documentation. The module reference is to be found :ref:`here <reference/modules:fibomat>`.

Please use the `issue system on gitlab <https://gitlab.com/viggge/fib-o-mat/-/issues>`__ for bug reports and questions concerning the package.

Made with |:black_heart:| and |:coffee:| at `HZB <https://www.helmholtz-berlin.de/>`__ and `FBH <https://www.fbh-berlin.de/en/>`__ in Berlin.

If you use this library in your work, please cite

Deinhart, V., Kern, L.-M., Kirchhof, J. N., Juergensen, S., Sturm, J., Krauss, E., Feichtner, T., Kovalchuk, S., Schneider, M., Engel, D., Pfau, B., Hecht, B., Bolotin, K. I., Reich, S., & Höflich, K. (2021). The patterning toolbox FIB-o-mat: Exploiting the full potential of focused helium ions for nanofabrication. Beilstein Journal of Nanotechnology, 12(1), 304–318. https://doi.org/10.3762/bjnano.12.25


.. toctree::
    :maxdepth: 2
    :glob:
    :hidden:

    Getting started <getting_started>
    User Guide <user_guide/user_guide>
    Use cases <use_cases/introduction>
    License <license>
    Contributors <contributors>
    Changelog <changelog>
    Module Reference <reference/modules>
