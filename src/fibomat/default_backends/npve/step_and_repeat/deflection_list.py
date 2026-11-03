"""Provides the :class:`DeflectionList` class, the points of a deflection list of an NPVE shape.

A deflection list is either embedded in the XML file or stored in a binary file next to it. The binary file (all values
little endian, 32 bit) consists of

* a header: the unsigned integers ``1`` (the meaning is unknown, all known files contain 1) and the number of points,
* for each point: the floats ``x`` and ``y`` (in µm) and the unsigned integer ``repeat``.

The dwell time of a point is ``repeat * DwellTime`` of the mill of the shape.

Example:
    >>> import numpy as np
    >>> points = DeflectionList(np.array([[0., 1.], [2., 3.]]), np.array([1., 2.]), shape_id=1, embedded=False)
    >>> len(points.to_bytes())
    32
"""
from __future__ import annotations

import numpy as np


__all__ = ['DeflectionList']


_POINT_TYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('repeat', '<u4')])
_HEADER_TYPE = np.dtype([('version', '<u4'), ('count', '<u4')])

_MAX_REPEAT = 2 ** 32 - 1

_DEFAULT_FILE_STEM = 'deflection_list'


class DeflectionList:
    """The points of a deflection list.

    The positions are stored with single precision, as in the binary files, so an embedded list and a list in a binary
    file contain the same numbers.
    """

    def __init__(self, positions: np.ndarray, repeats: np.ndarray, shape_id: int, embedded: bool):
        """
        Args:
            positions (np.ndarray): positions in µm, shape (n, 2)
            repeats (np.ndarray): integer multiples of the dwell time, shape (n,)
            shape_id (int): id of the shape, it makes the name of the binary file unique
            embedded (bool): if True, the list is written to the XML file, otherwise to a binary file

        Raises:
            ValueError: Raised if the list is empty, or a position is not finite, or a repeat value is no positive
                integer.
        """
        if len(positions) == 0:
            raise ValueError('A deflection list needs at least one point.')
        if not np.all(np.isfinite(positions)):
            raise ValueError('The positions of a deflection list must be finite.')
        if np.any(repeats != np.round(repeats)) or np.any(repeats < 1) or np.any(repeats > _MAX_REPEAT):
            raise ValueError(
                'The weights of the points of a deflection list must be positive integers (they are multiples of the '
                'dwell time of the mill). Choose a smaller dwell time for the mill and scale the weights.'
            )

        self._points = np.empty(len(positions), dtype=_POINT_TYPE)
        self._points['x'] = positions[:, 0]
        self._points['y'] = positions[:, 1]
        self._points['repeat'] = repeats

        self._shape_id = shape_id
        self._file_stem = _DEFAULT_FILE_STEM
        self.embedded = bool(embedded)

    def set_file_stem(self, file_stem: str) -> None:
        """Set the first part of the name of the binary file.

        Args:
            file_stem (str): the name is ``<file_stem>_<id of the shape>.dat``
        """
        self._file_stem = file_stem

    @property
    def data_filename(self) -> str:
        """Name of the binary file.

        Access:
            get
        """
        return f'{self._file_stem}_{self._shape_id}.dat'

    @property
    def n_points(self) -> int:
        """Number of points.

        Access:
            get
        """
        return len(self._points)

    def to_bytes(self) -> bytes:
        """The content of the binary file.

        Returns:
            bytes
        """
        header = np.array([(1, self.n_points)], dtype=_HEADER_TYPE)
        return header.tobytes() + self._points.tobytes()

    @property
    def center_offset(self) -> dict:
        """Value of the XML element ``CenterOffset``.

        Access:
            get
        """
        return {'x': 0.0, 'y': 0.0}

    @property
    def points_list(self) -> dict:
        """Value of the XML element ``PointsList`` (the points or the name of the binary file).

        Access:
            get
        """
        data = {
            'units': 0,  # µm
            'raw_bounding_box': {
                'x_min': float(self._points['x'].min()), 'x_max': float(self._points['x'].max()),
                'y_min': float(self._points['y'].min()), 'y_max': float(self._points['y'].max()),
            },
            'count': self.n_points,
            'rep_count': self.n_points,
        }
        if self.embedded:
            data['sub_list'] = {
                'points': [
                    {'x': float(point['x']), 'y': float(point['y']), 'repeat': int(point['repeat'])}
                    for point in self._points
                ]
            }
        else:
            data['data_filename'] = self.data_filename
        return data
