from typing import Sequence, Any

import numpy as np
import pint

from fibomat.linalg.vectors.vector_base import VectorBase, FloatTypes


class Vector(VectorBase[float]):
    @staticmethod
    def _validate_type(val: Any) -> bool:
        return isinstance(val, FloatTypes)

    @staticmethod
    def _build_np_array(val: Sequence[float]) -> np.ndarray:
        return np.array(val, dtype=float)

    @staticmethod
    def _make_zero_vector():
        return [0., 0.]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __mul__(self, other: Any):
        if isinstance(other, pint.Unit):
            from .dim_vector import DimVector
            return DimVector(self._vector * other)
        else:
            return super().__mul__(other)
