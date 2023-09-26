"""Provide the :class:`Mill` class."""
from typing import Optional

from fibomat.units import QuantityType, has_time_dim, Q_
from fibomat.mill.ionbeam import IonBeam


class MillBase:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return ('{}(' + ', '.join([key + '={}' for key in self._kwargs.keys()]) + ')').format(
            self.__class__.__name__, *self._kwargs.values()
        )

    # def __getattr__(self, item):
    #     return self.__getitem__(item)
    #
    # def __setattr__(self, key, value):
    #     if not key.startswith('_'):
    #         raise TypeError(
    #             "'MillBase' object does not support item assignment (if you are a developer, use variables with "
    #             " _ [underscore] as first character. Then, this check is bypassed)"
    #         )
    #     super().__setattr__(key, value)

    # def __getattribute__(self, item: str):
    #     print('__getattribute__', item)
    #     if not item.startswith('_') and item not in self._kwargs:
    #         raise AttributeError
    #     return object.__getattribute__(self, item)

        #
    # def __setattr__(self, key, value):
    #     raise NotImplementedError

    def __getitem__(self, item: str):
        try:
            value = self._kwargs[item]

            if value is None:
                raise KeyError('value is none')

            return value
        except KeyError as key_error:
            raise KeyError(
                f'Mill object does not have property "{item}". '
                'Maybe you need a SpecialMill with custom properties for the used backend?'
            ) # from key_error


class Mill(MillBase):
    """The `Mill` class is used to specify the, dwell_time per spot and the number of repats for a pattern.

    Optionally, the class can hold an object describing the shape of the ion beam which is needed if any kind of
    optimization is done.
    """
    def __init__(self, dwell_time: QuantityType, repeats: int):
        """
        Args:
            dwell_time (QuantityType): dwell time per spot
            repeats (int): number of repeats
        """

        if not isinstance(dwell_time, Q_):
            raise TypeError('dwell_time must be a quantity.')

        if not has_time_dim(dwell_time):
            raise ValueError('dwell_time must have dimension [time].')

        if not isinstance(repeats, int):
            raise TypeError('repats must be an int')

        if repeats < 1:
            raise ValueError('repeats must be at least 1.')

        super().__init__(dwell_time=dwell_time, repeats=repeats)

    @property
    def dwell_time(self) -> QuantityType:
        return self['dwell_time']

    @property
    def repeats(self) -> int:
        return self['repeats']


class SpecialMill(MillBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

