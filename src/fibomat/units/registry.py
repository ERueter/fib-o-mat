"""The shared pint unit registry of the package, including the custom ions and dose definitions.

pint quantities of different registries do not interoperate, so only :data:`ureg` is used throughout the package.

Custom definitions:

* ``ions = elementary_charge``: ``1 ions`` is exactly one elementary charge (only charges of +1 are of interest), so
  ``(1 * ureg.coulomb).to('ions')`` is identical to ``1 coulomb / elementary_charge`` (about 6.24e18 ions).
* ``[spotdose] = [current] * [time]``: charge (ions) per spot, e.g. ``ions``.
* ``[linedose] = [current] * [time] / [length]``: charge per length, e.g. ``ions / nm``.
* ``[areadose] = [current] * [time] / [length] ** 2``: charge per area, e.g. ``ions / nm**2``.
* The prefix ``micro-`` and the unit ``micron`` are written with the micro sign ``µ`` (U+00B5), e.g. ``µm``, with every
  version of pint (pint 0.26 changed the default to the Greek letter ``μ``, U+03BC). Both characters are accepted as
  input. Thereby, file names and file headers (e.g. of the FEI and spot list backends) do not depend on the version of
  pint.

.. note:: ``ions`` are a charge, not a count of ``[current] * [time] ** 2``: ``1 ions * s`` is *not* a spot dose.
"""
from __future__ import annotations

import logging

import pint  # type: ignore


__all__ = ['ureg']


ureg: pint.UnitRegistry = pint.UnitRegistry(system='mks')
"""pint unit registry."""

# write micro with the micro sign µ (U+00B5), as pint did before version 0.26, which prefers the Greek mu μ (U+03BC).
# The first symbol of a definition is used for formatting. pint logs redefinitions, which is not of interest here.
_pint_logger = logging.getLogger('pint')
_level = _pint_logger.level
_pint_logger.setLevel(logging.ERROR)
try:
    ureg.define('micro- = 1e-6 = µ- = μ- = u- = mu- = mc-')
    ureg.define('micron = micrometer = µ = μ')
finally:
    _pint_logger.setLevel(_level)

# add unit `ions` to the unit registry: 1 ions == 1 elementary_charge
ureg.define('ions = elementary_charge')

# define doses as derived dimensions
ureg.define('[spotdose] = [current] * [time]')
ureg.define('[linedose] = [current] * [time] / [length]')
ureg.define('[areadose] = [current] * [time] / [length] ** 2')
