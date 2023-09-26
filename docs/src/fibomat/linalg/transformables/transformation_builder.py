"""Provides the functions :func:`translate`, :func:`rotate`, :func:`scale` and :func:`mirror` to build chained
transformations.
"""
from __future__ import annotations

from typing import List, Optional, Union, Generic, TypeVar

from fibomat.linalg.vectors import VectorLike, Vector, DimVector

VectorT = TypeVar('VectorT', Vector, DimVector)


class _TransformationBuilder(Generic[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self):
        self._transformations: List[_TransformationBuilder[VectorT]] = [self]

    @property
    def transformations(self) -> List[_TransformationBuilder[VectorT]]:
        """Saved transformations.

        Access:
            get

        Returns:
            List[_TransformationBuilder]
        """
        return self._transformations

    def __or__(self, other: _TransformationBuilder[VectorT]) -> _TransformationBuilder[VectorT]:
        if not isinstance(other, _TransformationBuilder):
            raise TypeError('other is not an TransformationBuilder object.')

        self._transformations.append(other)
        return self


class _TranslationBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, trans_vec: VectorT):
        super().__init__()
        self.trans_vec = trans_vec


class _RotationBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, theta: float, origin: Optional[Union[VectorT, str]] = None):
        super().__init__()
        self.theta = theta
        self.origin = origin


class _ScaleBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, fac: float, origin: Optional[Union[VectorT, str]] = None):
        super().__init__()
        self.fac = fac
        self.origin = origin


class _MirrorBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, mirror_plane: VectorT):
        super().__init__()
        self.mirror_plane = mirror_plane


def translate(trans_vec: VectorT):
    """Translate a object by `trans_vec`.

    Args:
        trans_vec (VectorLike): translation vector

    Returns:
        _TranslationBuilder
    """
    return _TranslationBuilder(trans_vec)


def rotate(theta: float, origin: Optional[Union[VectorT, str]] = None):
    """Rotate a object `origin` with angle `theta` in math. positive direction (counterclockwise).

    Args:
        theta (float): rotation angle in rad
        origin (Vector, str, optional):
            origin of rotation. If not set, (0,0) is used as origin. If origin == 'center', the
            :attr:`Transformable.center` of the object will be used. The same applies for the case that
            origin == 'origin' with the :attr:`Transformable.origin` property. Default to None.

    Returns:
        _RotationBuilder
    """
    return _RotationBuilder(theta, origin)


def scale(fac: float, origin: Optional[Union[VectorT, str]] = None):
    """Scale a object homogeneously about `origin` with factor `fac`.

    Args:
        fac (float): rotation angle in rad
        origin (Optional[Union[linalg.VectorLike, str]], optional):
            origin of rotation. If not set, (0,0) is used as
            origin. If origin == 'center', the
            :attr:`Transformable.center` of the object will
            be used. The same applies for the case that
            origin == 'origin' with the
            :attr:`Transformable.origin` property. Default
            to None.

    Returns:
        _ScaleBuilder
    """
    return _ScaleBuilder(fac, origin)


def mirror(mirror_plane: VectorT):
    """Mirror a object mirrored about `mirror_plane`.

    Args:
        mirror_plane (VectorLike): mirror plane to be used.

    Returns:
        Transformable
    """
    return _MirrorBuilder(mirror_plane)
