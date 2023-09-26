"""Provides the :class:`RasterizedPoints` class."""
from __future__ import annotations
from typing import List, Optional

import numpy as np

from fibomat.linalg import Vector, VectorLike, BoundingBox
from fibomat.shapes import Shape


class RasterizedPoints(Shape):
    """Class represents pre-rasterized points."""
    def __init__(self, dwell_points: np.ndarray, is_closed: bool, description: Optional[str] = None):
        """
        Args:
            dwell_points (np.ndarray):
                dim must be 2 with shape = (-1, 3). First to components are coordinates and  third a dwell time
                multiplicand.
            is_closed (bool): if True, shape is considered as closed.
            description (str, optional): description

        Raises:
            ValueError: Raised if dimension or shape of dwell points is wrong
        """
        super().__init__(description)

        if dwell_points.ndim != 2:
            raise ValueError('dwell_points mus have dimension 2.')
        if dwell_points.shape[1] != 3:
            raise ValueError('dwell_points must have shape (N, 3)')

        self._dwell_points = dwell_points
        self._is_closed = bool(is_closed)

    @classmethod
    def merged(cls, other_raster_points: List[RasterizedPoints]) -> RasterizedPoints:
        """Create merged :class:`RasterizedPoints` class from list of :class:`RasterizedPoints`.

        Args:
            other_raster_points: points to be merged.

        Returns:
            RasterizedPoints
        """
        total_points = 0
        for raster_points in other_raster_points:
            total_points += raster_points.n_points

        dwell_points = np.empty(shape=(total_points, 3), dtype=float)

        i_offset = 0
        for raster_points in other_raster_points:
            dwell_points[i_offset:i_offset+raster_points.n_points] = raster_points.dwell_points
            i_offset += raster_points.n_points

        return cls(dwell_points, False)

    @property
    def positions(self):
        """Coordinates of dwell points.

        Access:
            get

        Returns:
            np.ndarray
        """
        view = self._dwell_points[:, :2]
        view.flags.writeable = False
        return view

    @property
    def weights(self):
        """Dwell time multiplicator of dwell points.

        Access:
            get

        Returns:
            np.ndarray
        """
        view = self._dwell_points[:, 2]
        view.flags.writeable = False
        return view

    @property
    def dwell_points(self):
        """Dwell points.

        Access:
            get

        Returns:
            np.ndarray
        """
        view = self._dwell_points[:]
        view.flags.writeable = False
        return view

    @property
    def n_points(self) -> int:
        """Number of dwell points.

        Access:
            get

        Returns:
            int
        """
        return len(self._dwell_points)

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def __repr__(self) -> str:
        return 'RasterizedPoints(...)'

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox.from_points(self.positions)

    @property
    def center(self) -> Vector:
        return Vector(np.mean(self.positions, axis=1))

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._dwell_points[:, :2] += trans_vec

    def _impl_rotate(self, theta: float) -> None:
        # pylint: disable=invalid-name
        theta = float(theta)
        m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        self._dwell_points[:, :2] = self._dwell_points[:, 2] @ m.T

    def _impl_scale(self, fac: float) -> None:
        self._dwell_points[:, :2] *= float(fac)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        # pylint: disable=invalid-name
        mirror_axis = Vector(mirror_axis)

        lx, ly = mirror_axis
        mirror_matrix = np.array([[lx*lx - ly*ly, 2*lx*ly], [2*lx*ly, ly*ly - lx*lx]]) / mirror_axis.length

        self._dwell_points[:, :2] = mirror_matrix.dot(self._dwell_points[:, :2].T)

    def repeats_applied(self, repeats: int) -> RasterizedPoints:
        """Return :class:`RasterizedPoints` with dwel points repeated `repeats` times.

        Args:
            repeats: number of repeats.

        Returns:
            RasterizedPoints
        """
        if repeats != 1:
            # TODO: check and replace
            # np.tile(self._dwell_points, (repeats, 1))
            return RasterizedPoints(np.concatenate([self._dwell_points]*repeats), self._is_closed)

        return self
