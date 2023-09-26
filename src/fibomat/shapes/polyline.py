"""Provide the :class:`Polyline` class."""
from __future__ import annotations
from typing import Optional, List

import numpy as np

from fibomat.shapes import shape
from fibomat.shapes.arc_spline import ArcSpline, ArcSplineCompatible
from fibomat.linalg import VectorLike, Vector, BoundingBox


class Polyline(shape.Shape, ArcSplineCompatible):
    """Polyline"""

    def __init__(self, points: List[VectorLike], description: Optional[str] = None, _closed: bool = False):
        """
        Args:
            points (VectorArrayLike): polyline points
            description (str, optional): description
        """
        super().__init__(description)

        # TODO: error checking?
        self._polygon = ArcSpline([(*vec, 0.) for vec in points], _closed)

    def to_arc_spline(self) -> ArcSpline:
        return self._polygon.clone_with_new_description(self.description)

    @property
    def points(self) -> np.ndarray:
        """Polyline points.

        Access:
            get

        Returns:
            np.ndarray
        """
        return self._polygon.vertices[:, :2]

    @property
    def n_points(self) -> int:
        """Number of points.

        Access:
            get

        Returns:
            int
        """
        return len(self._polygon)

    # def clone(self) -> Polyline:
    #     return self.__class__(self._points.clone())
    #
    # __copy__ = clone

    def __repr__(self) -> str:
        if self.n_points < 5:
            return '{}(points={!r})'.format(
                self.__class__.__name__, self.points)
        else:
            return '{}(points=...)'.format(
                self.__class__.__name__)

    @property
    def bounding_box(self) -> BoundingBox:
        return self._polygon.bounding_box

    @property
    def is_closed(self) -> bool:
        return self._polygon.is_closed

    @property
    def center(self) -> Vector:
        return self._polygon.center

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._polygon._impl_translate(trans_vec)  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._polygon._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        self._polygon._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._polygon._impl_mirror(mirror_axis)  # pylint: disable=protected-access
