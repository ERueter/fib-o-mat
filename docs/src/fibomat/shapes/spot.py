"""Provides the :class:`Spot` class."""
from __future__ import annotations
from typing import Optional

from fibomat.linalg import VectorLike, Vector, BoundingBox
from fibomat.shapes import shape


class Spot(shape.Shape):
    """0-dim spot."""

    def __init__(self, position: Optional[VectorLike] = None, *, description: Optional[str] = None):
        """
        Args:
            position (VectorLike, optional): position of the spot, default to (0, 0)
            description (str, optional): description
        """
        super().__init__(description)

        self._position: Vector = Vector(position) if position is not None else Vector()

    def __repr__(self) -> str:
        return '{}(position={!r})'.format(
            self.__class__.__name__, self._position)

    @property
    def position(self) -> Vector:
        """Position of the spot. Same as :attr:`center`

        Access:
            get

        Returns:
            Vector
        """
        return self._position

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self._position, self._position)

    @property
    def is_closed(self) -> bool:
        raise NotImplementedError

    @property
    def center(self) -> Vector:
        return self._position

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._position += trans_vec

    def _impl_rotate(self, theta: float) -> None:
        self._position = self._position.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        self._position *= fac

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._position = self._position.mirrored(mirror_axis)
