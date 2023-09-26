from typing import Type

from fibomat.linalg import (
    Transformable, DimTransformable, BoundingBox, DimBoundingBox, VectorLike, Vector, DimVectorLike, DimVector
)
from fibomat.units import U_
from fibomat.shapes import DimShape


class MocKTransformable(Transformable):
    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox((0, 0), (0, 0))

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        pass

    def _impl_rotate(self, theta: float) -> None:
        pass

    def _impl_scale(self, fac: float) -> None:
        pass

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        pass

    @property
    def center(self) -> Vector:
        return Vector(5, 6)


class MocKDimTransformable(DimTransformable):
    @property
    def bounding_box(self) -> DimBoundingBox:
        return DimBoundingBox((0, 0) * U_('µm'), (0, 0) * U_('µm'))

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        pass

    def _impl_rotate(self, theta: float) -> None:
        pass

    def _impl_scale(self, fac: float) -> None:
        pass

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        pass

    @property
    def center(self) -> DimVector:
        return Vector(5, 6) * U_('µm')


class MockDimShape(DimShape):
    def __init__(self):
        super().__init__(MocKDimTransformable(), U_('m'))

    @property
    def bounding_box(self) -> DimBoundingBox:
        return DimBoundingBox((0, 0) * U_('µm'), (0, 0) * U_('µm'))

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        pass

    def _impl_rotate(self, theta: float) -> None:
        pass

    def _impl_scale(self, fac: float) -> None:
        pass

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        pass

    @property
    def center(self) -> DimVector:
        return Vector(5, 6) * U_('µm')

