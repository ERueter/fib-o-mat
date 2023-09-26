from __future__ import annotations
from typing import Iterable, Any

import numpy as np

from fibomat.linalg.boundingboxes.boundingbox_base import BoundingBoxBase
from fibomat.linalg.vectors import DimVector, DimVectorLike
from fibomat.units import LengthQuantity, scale_factor, U_


class DimBoundingBox(BoundingBoxBase[DimVector, LengthQuantity]):
    _VectorClass = DimVector

    @classmethod
    def from_points(cls, points: Iterable[DimVector]) -> DimBoundingBox:
        """
        Constructs rectangular bounding box containing all `points`

        Args:
            points (VectorArrayLike): points which should be included in bounding box

        Returns:
            (BoundingBox): new `BoundingBox`
        """
        point_iter = iter(points)

        try:
            first_point = DimVector(next(point_iter))
        except StopIteration as stop_exception:
            raise ValueError('points must contain at least one point') from stop_exception

        unit = first_point.unit
        points_scaled = [first_point.vector]

        for point in point_iter:
            dim_point = DimVector(point)
            points_scaled.append((dim_point * scale_factor(unit, dim_point.unit)).vector)

        points = np.asarray(points_scaled)

        return DimBoundingBox(np.min(points, axis=0) * unit, np.max(points, axis=0) * unit)
