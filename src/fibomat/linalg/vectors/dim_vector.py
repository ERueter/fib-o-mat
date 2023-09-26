from typing import Sequence, Any

import numpy as np

from fibomat.linalg.vectors.vector_base import VectorBase, FloatTypes, SelfT
from fibomat.linalg.vectors.vector import Vector
from fibomat.units import Q_, U_, scale_factor


class DimVector(VectorBase[Q_]):
    @staticmethod
    def _validate_type(val: Any) -> bool:
        return isinstance(val, Q_) and val.u.dimensionality == '[length]' and isinstance(val.m, FloatTypes)

    @staticmethod
    def _build_np_array(val: Sequence[Q_]) -> np.ndarray:
        x: Q_ = val[0]
        y: Q_ = val[1]

        y.ito(x.u)

        return np.array([x.m, y.m], dtype=float) * x.u

    @staticmethod
    def _make_zero_vector():
        return [0*U_('µm'), 0*U_('µm')]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def vector(self) -> Vector:
        return Vector(self._vector.m)

    @property
    def unit(self) -> U_:
        return self._vector.u

    def normalized(self: SelfT) -> SelfT:
        """Create a new vector with same :attr:`Vector.phi` but :attr:`Vector.r` = 1.

        Returns:
            SelfT
        """
        return self.__class__(self._vector / np.linalg.norm(self._vector.m))

    def vector_as(self, unit: U_) -> Vector:
        """Returns a scaled vector in units of `unit`.

        Args:
            unit:

        Returns:
            Vector
        """
        return scale_factor(unit, self._vector.u) * Vector(self._vector.m)
