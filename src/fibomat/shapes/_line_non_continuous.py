"""Provides the :class:`Line` class."""
from __future__ import annotations
from typing import Optional, List

from fibomat.linalg import Vector, VectorLike, BoundingBox
from fibomat.shapes import shape
from fibomat.shapes.line import Line


class LineNonContinuous(shape.Shape):
    def __init__(self, line_segments: List[Line]):
        super().__init__()
        self._segments = line_segments

    def __repr__(self):
        raise NotImplementedError

    @property
    def segments(self):
        return self._segments

    @property
    def is_closed(self) -> bool:
        return False

    @property
    def bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    @property
    def center(self) -> Vector:
        raise NotImplementedError

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        raise NotImplementedError

    def _impl_rotate(self, theta: float) -> None:
        raise NotImplementedError

    def _impl_scale(self, fac: float) -> None:
        raise NotImplementedError

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        raise NotImplementedError
