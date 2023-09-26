import numpy as np

from fibomat.units import LengthUnit, TimeUnit, has_length_dim, has_time_dim


class RasterizedPattern:
    """Class to represent dwell points (position + dwell time). In contrast to RasterizedPoints, actual dwell times
    are collected in this class."""

    def __init__(self, dwell_points: np.ndarray, length_unit: LengthUnit, time_unit: TimeUnit):
        """
        Args:
            dwell_points (np.ndarray):
                list of dwell pints with shape (n, 3). each point should contain (x, y, dwell_time).
            length_unit (LengthUnit): length unit of points
            time_unit (TimeUnit): time unit of dwell times
        """
        if dwell_points.ndim != 2 or dwell_points.shape[1] != 3:
            raise ValueError('dwell_points must be a list of shape (n, 3).')

        if not has_length_dim(length_unit):
            raise ValueError('lenght_unit must have dimension [length]')

        if not has_time_dim(time_unit):
            raise ValueError('lenght_unit must have dimension [time]')

        self._dwell_points = dwell_points
        self._length_unit = length_unit
        self._time_unit = time_unit

    @property
    def dwell_points(self):
        """List of dwell points (position + dwell time)

        Returns:
            np.ndarray

        Access:
            get
        """
        view = self._dwell_points[:]
        view.flags.writeable = False
        return view

    @property
    def positions(self):
        """List of positions

        Returns:
            np.ndarray

        Access:
            get
        """
        view = self._dwell_points[:, :2]
        view.flags.writeable = False
        return view

    @property
    def dwell_times(self):
        """List of dwell times

        Returns:
            np.ndarray

        Access:
            get
        """
        view = self._dwell_points[:, 2]
        view.flags.writeable = False
        return view

    @property
    def length_unit(self):
        """Length unit of positions

        Returns:
            LengthUnit

        Access:
            get
        """
        return self._length_unit

    @property
    def time_unit(self):
        """Time unit of dwell times

        Returns:
            TimeUnit

        Access:
            get
        """
        return self._time_unit
