Development
===========

|:test_tube:|  Contributing
---------------------------
To contribute custom code, follow the steps below.

    1. fork fib-o-mat
    2. create a new branch, e.g. ``git checkout -b my-new-branch``
    3. commit your changes, ``git add ...``, ``git commit -m "..."``
    4. push the code. ``git push origin my-new-branch``
    5. create a merge request from the fork (see `here <https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html#new-merge-request-from-a-fork>`__)

|:test_tube:| Versioning
------------------------
Versioning is done with help of `bump2version <https://github.com/c4urself/bump2version>`__.
Run

.. code-block:: bash

    $ bump2version {major|minor|patch}

in the root folder of fib-o-mat to increase the corresponding number. Push the resulting commit to the git repository.

|:test_tube:| Building the docs
-------------------------------

In the `[fib-o-mat]/docs` folder run

.. code-block:: bash

    $ make docs

The docs are outputted in `[fib-o-mat]/build/sphinx/html`.

For this, the some extra packages must be installed. Install them by running

    .. code-block:: bash

    $ pip install -r requirements_docs.txt

in the `[fib-o-mat]/docs` folder.

|:test_tube:| Building wheel packages (for pypi)
------------------------------------------------
Run

.. code-block:: bash

    $ python setup.py sdist

to build a source distribution which will be placed in `[fib-o-mat]/dist`.

Linux
+++++

Linux wheel packages are build with `dockcross <https://github.com/dockcross/dockcross>`__.
For this, `docker <https://www.docker.com/>`__ must be installed.

Run the build script

.. code-block:: bash

    $ ./build-linux.sh

The wheel packages for python 3.8 and 3.9 are placed in `[fib-o-mat]/dist`.

https://github.com/dockcross/python-manylinux-demo

Windows
+++++++

Run

.. code-block:: bash

    $ python setup.py bdist-wheel

on a MS Windows system with Visual Studio installed. It is test with with Visual Studio 2017.

The wheel packages for python 3.8 and 3.9 are placed in `[fib-o-mat]/dist`.
