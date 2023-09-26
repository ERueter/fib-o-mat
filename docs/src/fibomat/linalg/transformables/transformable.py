"""
Provides the :class:`Transformable` class.
"""
from __future__ import annotations
from typing import Optional, Union, Callable, TypeVar
import abc

from fibomat.linalg.vectors import Vector, VectorLike
from fibomat.linalg.transformables.transformation_builder import _TransformationBuilder
from fibomat.linalg.transformables.transformable_base import TransformableBase
from fibomat.linalg.boundingboxes import BoundingBox


class Transformable(TransformableBase[Vector, float, BoundingBox], abc.ABC):
    """
    :class:`Transformable` is a base class providing the translate, rotate and uniform scale
    transformations.

    In order to use this mixin in a child class, the following methods and properties must be implemented:
        * :attr:`Transformable.center`
        * :meth:`Transformable.translate`
        * :meth:`Transformable.simple_rotate`
        * :meth:`Transformable.simple_scale`

    """

    _VectorClass = Vector

    def __init__(self, description: Optional[str] = None):
        """
        Args:
            pivot (VectorLike, optional): if set, the :attr:`Transformable.pivot` is set to `pivot`. If not set,
                                           :attr:`Transformable.center` is used as default.

            description (str, optional): optional description
        """
        super().__init__(description=description)

