from typing import Union, Iterable

import numpy as np

from fibomat.linalg.vectors.vector import Vector
from fibomat.linalg.vectors.dim_vector import DimVector
from fibomat.linalg.vectors.vector_base import VectorValueError, FloatTypes

from fibomat.units import LengthQuantity


VectorLike = Union[
    Vector,
    Iterable[float]
]
"""vector-like objects for type hints"""

DimVectorLike = Union[DimVector, LengthQuantity]  # Tuple[VectorLike, LengthUnit]
"""dim-vector-like objects for type hints"""


def angle_between(vec_1: Union[VectorLike, DimVectorLike], vec_2: Union[VectorLike, DimVectorLike]) -> float:
    """Returns the the angle between two vectors.

    Args:
        vec_1 (VectorLike): first vector
        vec_2 (VectorLike): second vector

    Returns:
        float
    """
    vec_1 = np.asarray(vec_1)
    vec_2 = np.asarray(vec_2)

    if vec_1.shape != (2,) or vec_2.shape != (2,):
        raise ValueError('vec_1 and vec_2 must be VectorLike.')

    if not len(vec_1) == 2 or not len(vec_2):
        raise ValueError('vec_1 and vec_2 must have length 2.')

    return np.arccos(vec_1.dot(vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))


def signed_angle_between(vec_1: Union[VectorLike, DimVectorLike], vec_2: Union[VectorLike, DimVectorLike]):
    """Returns the signed the angle between two vectors.
    The angle is positive if the rotation from vec_1 to vec_2 is in positive direction
    using the smaller angle between them.

    https://stackoverflow.com/a/16544330

    Args:
        vec_1 (VectorLike): first vector
        vec_2 (VectorLike): second vector

    Returns:
        float
    """
    vec_1 = np.asarray(vec_1)
    vec_2 = np.asarray(vec_2)

    if vec_1.shape != (2,) or vec_2.shape != (2,):
        raise ValueError('vec_1 and vec_2 must be VectorLike.')

    dot = np.dot(vec_1, vec_2)
    det = np.linalg.det(np.c_[vec_1, vec_2])

    return np.arctan2(det, dot)
