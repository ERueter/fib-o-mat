from __future__ import annotations
from typing import Iterable, Any

import numpy as np

from fibomat.linalg.boundingboxes.boundingbox_base import BoundingBoxBase
from fibomat.linalg.vectors import Vector, VectorLike
from fibomat.units import U_


class BoundingBox(BoundingBoxBase[Vector, float]):
    _VectorClass = Vector

    @classmethod
    def from_points(cls, points: Iterable[VectorLike]) -> BoundingBox:
        """
        Constructs rectangular bounding box containing all `points`

        Args:
            points (VectorArrayLike): points which should be included in bounding box

        Returns:
            (BoundingBox): new `BoundingBox`
        """
        points = np.asarray(points)

        if len(points) == 0 or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must not be empty and contain vector compatible values.')

        return BoundingBox(np.min(points, axis=0), np.max(points, axis=0))

    def __mul__(self, other: Any):
        if isinstance(other, U_):
            from .dim_boundingbox import DimBoundingBox
            return DimBoundingBox(self._lower_left * other, self._upper_right * other)
        else:
            raise NotImplementedError


