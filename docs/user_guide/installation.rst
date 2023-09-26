Installation
============

Fib-o-mat requires python 3.8+ and can be easily installed via pip on the most systems. Pre-build packages are available for 64 bit
linux and windows systems. It is highly recommended to use a virtual environment, e.g.

.. code-block:: bash

    $ python -m venv .venv
    # for *nix systems
    $ source .venv/bin/activate
    # for MS windows
    $ .venv\Scripts\activate.bat
    # install fib-o-mat
    $ pip install --upgrade fibomat

This should run a usual GNU/Linux or Windows system. See `here <https://docs.python.org/3/library/venv.html>`__ for more
information on virtual environments.

If no suited pre-build packages are found, the pip command above triggers compilation of the package. This requires a c++ compiler supporting c++17 (e.g. gcc >= 8.3, VS 2017 15.3), cmake >= 3.14 and nodejs >= 10.13.0.

.. note:: macOS is currently not supported. Even so, the package might be build correctly and can used without any problems.

Building from source
--------------------
Clone the git repository with

.. code-block:: bash

    $ git clone https://gitlab.com/viggge/fib-o-mat

and run the following command in the fib-o-mat directory

.. code-block:: bash

    $ pip install .
    # or
    $ pip install -e .

The latter installs fib-o-mat in development mode.

.. note:: Due to a bug in the underlying build system, you have to remove the `_skbuild` folder manually if you try to install (a modified) version of fib-o-mat again. See also
 `here <https://github.com/scikit-build/scikit-build/issues/386>`__ for a bug report.
