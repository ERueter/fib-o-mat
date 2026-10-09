Units
=====

All physical quantities of the user-facing API (lengths, times, currents, doses, ...) carry a unit. The unit system is a thin layer on top of the `pint library <https://github.com/hgrecco/pint>`__ and lives in the :mod:`~fibomat.units` submodule.

The system has three parts:

    1. **unit tags** (:func:`~fibomat.units.unit`) create values,
    2. **dimensioned values** (:class:`~fibomat.units.DimFloat`) hold a number and a unit, and know their *dimension* (length, time, ...),
    3. **dimension checks** (:func:`~fibomat.units.check_units`) make sure that a function only gets values with the dimension it expects.

Creating values
---------------

A dimensioned value is created by multiplying a number with a unit tag. The unit tag is created with :func:`~fibomat.units.unit` from any unit string which pint understands ::

    from fibomat.units import unit

    length = 1.5 * unit('µm')
    dwell_time = 2 * unit('ms')
    current = 10 * unit('pA')
    dose = 10 * unit('ions / nm**2')

    um = unit('µm')  # a unit tag can be stored and reused
    other_length = 3 * um

The result is a :class:`~fibomat.units.DimFloat`. :func:`~fibomat.units.unit` is also available as ``U_`` (the name which older versions used for pint units).

The unit tag remembers the dimension of the unit. For the common units, the dimension is known statically (for type checkers like mypy and pyright) and at runtime:

    ===================== ==============================================================
    Dimension             Units known to :func:`~fibomat.units.unit`
    ===================== ==============================================================
    ``LengthDimension``   ``m``, ``cm``, ``mm``, ``µm`` (``um``), ``nm``, ``km``, ``in``, ``ft``, ...
    ``TimeDimension``     ``s``, ``ms``, ``µs`` (``us``), ``ns``, ``min``, ``h``, ``day``
    ``CurrentDimension``  ``A``, ``mA``, ``µA`` (``uA``), ``nA``, ``pA``, ``fA``
    ===================== ==============================================================

Other units work as well; they are just not tagged with a dimension statically. If you want a static dimension for such a unit, or if you want to validate a unit string, pass the dimension explicitly ::

    from fibomat.units import AreaDoseDimension, LengthDimension

    dose = 10 * unit('ions / nm**2', AreaDoseDimension)
    wrong = 1 * unit('kg', LengthDimension)  # raises DimensionError: 'kg' is not a length

Dimensions
----------

A dimension is a class (a *tag*) which stands for a pint dimensionality. The following tags are available:

    ============================== ============================================ ==========================
    Tag                            Pint dimensionality                          Example
    ============================== ============================================ ==========================
    ``LengthDimension``            ``[length]``                                 ``1 * unit('µm')``
    ``TimeDimension``              ``[time]``                                   ``1 * unit('ms')``
    ``CurrentDimension``           ``[current]``                                ``1 * unit('pA')``
    ``CurrentDensityDimension``    ``[current] / [length] ** 2``                ``1 * unit('pA / µm**2')``
    ``SpotDoseDimension``          ``[current] * [time]``                       ``1 * unit('ions')``
    ``LineDoseDimension``          ``[current] * [time] / [length]``            ``1 * unit('nC / µm')``
    ``AreaDoseDimension``          ``[current] * [time] / [length] ** 2``       ``1 * unit('ions / nm**2')``
    ``DimensionlessDimension``     no dimension                                 ``1 * unit('')``
    ============================== ============================================ ==========================

The unit ``ions`` is defined as one elementary charge (only ions with a charge of +1 are of interest), so doses can be given in ions as well as in coulomb.

A tag is used as the generic parameter of :class:`~fibomat.units.DimFloat` in type hints, e.g. ``DimFloat[LengthDimension]``.

Calculating with values
-----------------------

Dimensioned values can be added, subtracted and compared if they have the same dimension. The units may differ, the result has the unit of the left operand. They can be multiplied and divided by numbers and by other dimensioned values ::

    a = 1.5 * unit('µm')
    b = 500 * unit('nm')

    a + b  # 2.0 µm
    a - b  # 1.0 µm
    a * 2  # 3.0 µm
    a / b  # 0.003 µm / nm  (the result of a division by a value with another unit has a compound unit)
    a / (2 * unit('ms'))  # 0.75 µm / ms
    a < b  # False

Adding values with different dimensions (e.g. a length and a time) raises a pint ``DimensionalityError``.

Converting values
-----------------

To get the number of a value in a given unit, use :meth:`~fibomat.units.DimFloat.m_as` or :func:`~fibomat.units.scale_to`; both return a plain float. :meth:`~fibomat.units.DimFloat.to` returns a new dimensioned value ::

    from fibomat.units import scale_factor, scale_to

    a.m_as('nm')  # 1500.0 (float)
    scale_to(unit('nm'), a)  # the same
    a.to('nm')  # a DimFloat with the unit nm
    a.magnitude, a.units  # the number and the unit (as stored)

    scale_factor(unit('µm'), unit('nm'))  # 0.001: multiply a number in nm to get µm

.. note:: Conversions can have the usual floating point rounding errors, e.g. ``(1.5 * unit('µm')).m_as('nm')`` is ``1499.9999999999998``. Do not compare converted values with ``==``.

Dimensioned vectors, bounding boxes and shapes use the same units: ``Vector(1, 2) * unit('µm')`` is a :class:`~fibomat.linalg.DimVector` (see :doc:`preliminary`) and ``Line((0, 0), (1, 1)) * unit('µm')`` a :class:`~fibomat.shapes.DimShape`. A tuple cannot be multiplied with a unit, use ``Vector`` instead.

Checking dimensions
-------------------

Type hints with a dimension tell type checkers and readers which dimension a function expects. Type checkers cannot verify values which are created at runtime, though. Therefore, the decorator :func:`~fibomat.units.check_units` validates the *actual* dimension of annotated arguments and return values whenever the function is called ::

    from fibomat.units import DimFloat, DimensionError, LengthDimension, check_units

    @check_units
    def double(distance: DimFloat[LengthDimension]) -> DimFloat[LengthDimension]:
        return 2 * distance

    double(4 * unit('nm'))  # ok
    double(4 * unit('s'))  # DimensionError: 'distance': expected dimension '[length]' (LengthDimension), got [time]
    double(4.)  # DimensionError: 'distance': expected a DimFloat[LengthDimension], got float

The decorator checks parameters and the return value which are annotated as ``DimFloat[...]`` (or any other dimensioned type, e.g. a dimensioned vector). ``Optional[...]``, ``Union[...]`` and ``Tuple[...]`` of such annotations are checked as well; ``None`` is accepted for an ``Optional`` ::

    import typing as t

    @check_units
    def func(
        pitch: t.Optional[DimFloat[LengthDimension]] = None,
        size: t.Tuple[DimFloat[LengthDimension], DimFloat[LengthDimension]] = (1 * unit('µm'), 1 * unit('µm')),
    ) -> None:
        ...

Parameters without such an annotation are not touched. The annotations are evaluated on the first call of the function.

The functions and classes of fib-o-mat which take dimensioned values are decorated in this way. A value without a unit (e.g. ``Curve(0.5, ...)`` instead of ``Curve(0.5 * unit('µm'), ...)``) or with a wrong dimension raises a :class:`~fibomat.units.DimensionError`, which is a ``TypeError``. Values with the right dimension but a wrong value (e.g. a negative pitch) raise a ``ValueError``.

Custom dimensions
-----------------

New dimensions are defined by deriving from :class:`~fibomat.units.Dimension` and naming the pint dimensionality ::

    from fibomat.units import Dimension

    class ForceDimension(Dimension):
        pint_dimension = '[force]'

    force = 4 * unit('N', ForceDimension)  # DimFloat[ForceDimension]
    ForceDimension.matches(force.quantity)  # True

Custom units and dimensions of the dose family are defined in the pint registry :data:`fibomat.units.ureg` (see the module documentation for details).

.. note:: Older versions of fib-o-mat used ``Q_`` for quantities (``Q_('1 ms')``). ``Q_`` is deprecated; use ``1 * unit('ms')``.

The code of the basic examples can be found in `examples/units.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/units.py>`__.
