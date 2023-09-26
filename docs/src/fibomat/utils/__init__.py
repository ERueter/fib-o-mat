"""Provide some utility functions."""

import numpy as np

from fibomat.utils.pathlike import PathLike
from fibomat.utils.math import float_gcd, mod_2pi


def make_read_only_view(array: np.ndarray):
    view = array


__all__ = ['PathLike', 'float_gcd', 'mod_2pi']
