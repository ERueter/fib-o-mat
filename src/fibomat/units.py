"""This submodule implements (physical) unit support for the package (units = physical units).

It uses the python module `pint`. For usage, see the example below or the
documentation at https://pint.readthedocs.io/en/0.10.1/ ::

    from fibomat.units import Q_, U_

    # factor of 1. exists implicitly
    a = 1. * Q_('µm') = Q_('1. µm') = 1. * U_('µm')

    a_in_m = a.to('m')

    a_raw = a.magnitude
    a_unit = a.units

    dose = 12 * Q('ions / nm**2')
    dose_in_c_per_um2 = dose.to('coulomb / µm**2')
"""
from typing import Union, NewType

import pint  # type: ignore


ureg: pint.registry.UnitRegistry = pint.UnitRegistry(system='mks')
"""pint unit registry"""

# add unit `Ions` to the unit registry
# we are only interested in charges +1, so it is fine
ureg.define('ions = elementary_charge')  # = ions ?

# define doses as derived dimensions
ureg.define('[spotdose] = [current] * [time]')
ureg.define('[linedose] = [current] * [time] / [length]')
ureg.define('[areadose] = [current] * [time] / [length] ** 2')


Q_ = ureg.Quantity
"""Quantity constructor."""


U_ = ureg.Unit
"""Unit constructor"""


UnitType = Union[pint.Unit]  # NewType('UnitType', pint.Unit)
"""Generic pint unit type used for typing annotation. Objects constructed with :attr:`U_` hae this type."""


# https://github.com/python/mypy/issues/6701
# class PintUnit:
#     def __getattr__(self, item: str) -> Any:
#         pass


LengthUnit = pint.Unit  # NewType('LengthUnit', UnitType)
TimeUnit = pint.Unit  # NewType('TimeUnit', UnitType)
# LengthUnit = NewType('LengthUnit', UnitType)


QuantityType = Union[pint.Quantity]  # NewType('QuantityType', pint.Quantity)
"""pint quantity type used for typing annotation"""

LengthQuantity = QuantityType  # NewType('LengthQuantity', QuantityType)
TimeQuantity = QuantityType  # NewType('TimeQuantity', QuantityType)


def has_length_dim(quant_or_unit: Union[UnitType, QuantityType]) -> bool:
    """Check if `quant_or_unit` has dimension `[length]`.

    Args:
        quant_or_unit (UnitType, QuantityType): unit or quantity to be checked.

    Returns:
        bool
    """
    return quant_or_unit.dimensionality == '[length]'
    # return ureg.check(qunat, '')


def has_time_dim(quant_or_unit: Union[UnitType, QuantityType]) -> bool:
    """Check if `quant_or_unit` has dimension `[time]`.

    Args:
        quant_or_unit (UnitType, QuantityType): unit or quantity to be checked.

    Returns:
        bool
    """
    return quant_or_unit.dimensionality == '[time]'


def scale_factor(base_unit: Union[UnitType, QuantityType], other_unit: Union[UnitType, QuantityType]) -> float:
    """Calculates the scaling factor needed to convert `other_unit` to `base_untit`.

    .. warning:: If `base_unit` or `other_unit` are quantities, the magnitude is ignored.

    Args:
        base_unit (Union[UnitType, QuantityType]): base unit
        other_unit (Union[UnitType, QuantityType]): other unit

    Returns:
        float: scaling factor
    """
    base = base_unit if isinstance(base_unit, pint.Unit) else 1. * base_unit.units
    other_quant = 1. * other_unit if isinstance(other_unit, pint.Unit) else 1. * other_unit.units

    return other_quant.to(base).magnitude


def scale_to(base_unit: Union[UnitType, QuantityType], quantity: QuantityType) -> float:
    """Scales quantity `quantity` to `base_unit` and returns the result as `float`.

    Args:
        base_unit (Union[UnitType, QuantityType]): base unit
        quantity (QuantityType): quantity to be scaled

    Returns:
        float: scaled quantity
    """
    base = base_unit if isinstance(base_unit, pint.Unit) else base_unit.units
    return quantity.to(base).magnitude
