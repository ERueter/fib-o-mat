Preliminary
===========

Vectors
-------
fib-o-mat ships a :class:`~fibomat.linalg.vectors.vector.Vector` class which represent a 2-dim mathematical vector or point in euclidean space. A vector can be constructed in
various ways:

    - ``Vector()`` creates a null-vector
    - ``Vector(x=1, y=2) = Vector(1, 2)`` creates a cartesian vector with `(1, 2)` as components
    - ``Vector(np.array([1, 2]))`` creates the same vector as above from a numpy array
    - ``Vector([1, 2])`` creates the same vector as above from a list
    - ``Vector((1, 2))`` creates the same vector as above from a tuple
    - ``Vector(r=1, phi=np.pi)`` creates a vector from polar coordinates
    - ``Vector(Vector(1, 2))`` copies the passed vector object


:class:`~fibomat.linalg.vectors.vector.Vector` support the usual mathematical operations ::

    u = Vector(1, 2)
    v = Vector(3, 4)

    print(u + v)  # result in Vector(4, 6)
    print(u - v)  # result in Vector(-2, -2)
    print(4 * u)  # result in Vector(4, 8)
    print(u / 2)  # result in Vector(0.5, 1)

    print(u + (1, 2))  # results also in Vector(4, 6)
    print(u + [1, 2])  # results also in Vector(4, 6)
    print(u + np.array((1, 2)))  # results also in Vector(4, 6)

    print(u.dot(v)) # prints the dot product between u and v, in this case 9

Vectors can be accessed component-wise ::

    print(u.x, u.y)  # prints "1, 2"
    print(u[0], u[1])  # prints "1, 2", too. u[0] = u.x, u[1] = u.y

    w = Vector(r=1, phi=np.pi)
    print(u.r, u.phi)  # prints 1, 3.14159

.. warning:: Vector values cannot be changed by accessing its elements. E.g. ``u.x = 5`` is not working.
             To change a component, a new :class:`~fibomat.linalg.vector.Vector` must be constructed
             ``new_u = Vector(x=5, y=u.y)``

Further, some other properties of the vector can be accessed ::

    print(v.length)  # prints out the norm (length) of the vector
    print(v.angle_about_x_axis)  # prints the angle of the vector and the positive x-axis. the result will be in [0, 2pi]

    print(u.close_to(v))  # returns True, if u is nearly v and otherwise False.

    print(angle_between(u, v))  # prints the angle between u and v

Some other operations which can be applied to vectors ::

    u_rot = u.rotated(np.pi/2)  # rotated the vector counterclockwise by np.pi/2 around the origin.
    u_mir = u.mirrored([1, 0])  # mirrors the vector at the positive x-axis
    u_norm = u.normalized()  # returns a vector pointing in the same direction as `u` but with length = 1

:class:`~fibomat.linalg.vectors.vector.Vector` can be converted to a numpy array with ::

    np_array = np.asarray(u)

To describe dimensioned vectors, the :class:`~fibomat.linalg.vectors.dim_vector.DimVector` class exist.
This class supports the same function and methods as given above for the :class:`~fibomat.linalg.vectors.vector.Vector` class.

.. todo:: Add example here.


All the code snippets above are combined here: `<https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/vectors.py>`__.

.. note:: Not all methods on and with vectors are introduced above. Please consult the module reference for all available functions and methods.


Physical units
--------------

fibomat uses the `pint library <https://github.com/hgrecco/pint>`__ to represent physical units. All functionality is encapsulated in the :mod:`~fibomat.units` submodule.

A unit can be constructed with ::

    length_unit = U_('µm')
    dose_unit = U_('ions / nm**2')

Quantities can be defined nearly identical ::

    length = Q_('1 nm')
    dose = Q_('10 ions / nm**2')

     another_length = 10 * length_unit  # equal to 10 * U_('µm')

    # three version to create a dimensioned vector
    dim_vector = (3, 4) * U_('µm')
    dim_vector2 = Vector(3, 4) * U_('µm')
    dim_vector3 = DimVector(3 * U_('µm'), 4 * U_('µm'))

Quantities can be scaled and scale factors can be calculated ::

    length_in_um = scale_to(U_('µm'), length)  # NOTE: length_in_um is a float now and NOT a quantity anymore

    scale_factor = scale_factor(U_('µm'), U_('nm'))  # scale factor (float) to scale from nm to µm

Alternatively, all functions and methods defined in the pint module itself can be used ::

    length_in_um = length.m_as('µm')  # identical to scale_to(U_('µm'), length)
    # ...

.. foo
    Sometimes dimensioned objects are need, for example a :class:`~fibomat.linalg.vector.Vector` or a :class:`~fibomat.shapes.shape.Shape` with a length dimension.
    To express a vector with a length dimension, fibomat provides the :class:`~fibomat.linalg.dimvector.DimVector`. This class combines a vector and a length unit ::
        dim_vector = (Vector(1, 2), U_('µm')) = ((1, 2), U_('µm'))
    To express any other dimensioned object besides vectors, use the :class:`~fibomat.dimensioned_object.DimensionedObj` (or short :class:`~fibomat.dimensioned_object.DimObj`) class ::
        dim_spot = DimObj(Spot(1, 2), U_('µm')) = (Spot(1, 2), U_('µm'))
        dim_line = DimObj(Line(start=(0, 0), end=(1, 1), U_('mm')) = (Line(start=(0, 0), end=(1, 1), U_('mm'))
    Everywhere a :class:`~fibomat.linalg.dimvector.DimVector` or :class:`~fibomat.dimensioned_object.DimensionedObj` is needed, you can pass a tuple like the ones shown above as argument.

Examples given in: `<https://gitlab.com/viggge/fib-o-mat/-/blob/master/examples/units.py>`__.

Immutability
------------

All classes besides the :class:`~fibomat.sample.Sample`, :class:`~fibomat.site.Site` and the lattice builder classes are immutable.
This means, once an object is constructed, it cannot be changed or altered anymore.

For example, the x component of :class:`~fibomat.linalg.vectors.vector.Vector`  can be read but not set or the
:meth:`~fibomat.linalg.vectors.vector.Vector.rotated` method returns a
rotated :class:`~fibomat.linalg.vectors.vector.Vector`  but will not change the original one.
