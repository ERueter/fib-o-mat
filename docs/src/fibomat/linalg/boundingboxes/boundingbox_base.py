from __future__ import annotations
from typing import Generic, TypeVar, Iterable, Union, Type
import abc
import copy

import numpy as np

from fibomat.linalg.vectors import Vector, DimVector, FloatTypes, VectorValueError
from fibomat.units import LengthQuantity


VectorT = TypeVar("VectorT", Vector, DimVector)
ScalarT = TypeVar("ScalarT", int, float, np.integer, LengthQuantity)
SelfT = TypeVar("SelfT", bound="BoundingBoxBase")


class BoundingBoxBase(Generic[VectorT, ScalarT], abc.ABC):
    _VectorClass: Type[VectorT]

    def __init__(self, lower_left: np.ndarray, upper_right: np.ndarray):
        self._lower_left = self._VectorClass(lower_left)
        self._upper_right = self._VectorClass(upper_right)

        self._is_valid()

    @classmethod
    @abc.abstractmethod
    def from_points(cls, points: Iterable[VectorT]) -> BoundingBoxBase[VectorT]:
        """
        Constructs rectangular bounding box containing all `points`

        Args:
            points (VectorArrayLike): points which should be included in bounding box
            description (str, optional): description

        Returns:
            (BoundingBox): new `BoundingBox`
        """
        raise NotImplementedError

    def _is_valid(self):
        if (
            self._lower_left.x > self._upper_right.x
            or self._lower_left.y > self._upper_right.y
        ):
            raise ValueError("Invalid coordinates for bounding box")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBoxBase):
            raise NotImplementedError
        return self._lower_left.close_to(
            other._lower_left
        ) and self._upper_right.close_to(other._upper_right)

    def close_to(self, other: object) -> bool:
        return self == other

    def __repr__(self):
        return "BoundingBox(lower_left=({}, {}), upper_right=({}, {}))".format(
            self._lower_left.x,
            self._lower_left.y,
            self._upper_right.x,
            self._upper_right.y,
        )

    @property
    def corners(self) -> Iterable[VectorT]:
        return (
            self._lower_left,
            self._upper_right,
            self._VectorClass(self._lower_left.x + self.width, self._lower_left.y),
            self._VectorClass(self._lower_left.x, self._lower_left.y + self.height),
        )

    @property
    def lower_left(self) -> VectorT:
        """
        Vector: lower left coordinate

        Access:
            get
        """
        return self._lower_left

    @property
    def upper_right(self) -> VectorT:
        """
        Vector: upper richt coordinate

        Access:
            get
        """
        return self._upper_right

    @property
    def width(self) -> ScalarT:
        """
        float: with of bounding box

        Access:
            get
        """
        return self._upper_right[0] - self._lower_left[0]

    @property
    def height(self) -> ScalarT:
        """
        float: with of bounding box

        Access:
            get
        """
        return self._upper_right[1] - self._lower_left[1]

    @property
    def area(self) -> ScalarT:
        """
        float: area of bounding box

        Access:
            get
        """
        return self.width * self.height

    @property
    def center(self) -> VectorT:
        """
        Vector: center bounding box

        Access:
            get
        """
        return (self._upper_right + self._lower_left) / 2

    def clone(self: SelfT) -> SelfT:
        """
        Clones the bounding box

        Returns:
            (BoundingBox): cloned box
        """
        return copy.deepcopy(self)

    def scaled(self: SelfT, val: float) -> SelfT:
        """
        Return a scaled version of the `BoundingBox`.

        Args:
            val: scale factor

        Returns:
            BoundingBox
        """
        val = float(val) / 4

        shift = (self.width * val, self.height * val)

        return self.__class__(self._lower_left - shift, self._upper_right + shift)

    def overlaps_with(self, other: BoundingBoxBase) -> bool:
        """
        Checks if this bounding box overlaps with other.

        Args:
            other: other bounding box

        Returns:
            bool: True if self and other overlap
        """

        overlap_x = True
        overlap_y = True

        if (
            self._lower_left.x > other._upper_right.x
            or self._upper_right.x < other._lower_left.x
        ):
            overlap_x = False

        if (
            self._lower_left.y > other._upper_right.y
            or self._upper_right.y < other._lower_left.y
        ):
            overlap_y = False

        return overlap_x and overlap_y

    def contains(self, other: Union[BoundingBoxBase, VectorT]) -> bool:
        """
        Checks if this bounding box contains other. Alternatively, you can use `in` syntax. Note, that a box is contains
        always itself. ::

            box_1 = BoundingBox([1, 2], [3, 4])
            box_2 = BoundingBox([1, 3], [0, 4])

            print(box_1.contains(box_2)
            # or
            print(box_2 in box_1)

        Args:
            other (BoundingBox): other box

        Returns:
            (bool): True if `self` contains `other`
        """
        if isinstance(other, self.__class__):
            if (
                self._lower_left.x <= other._lower_left.x
                and self._lower_left.y <= other._lower_left.y
                and self._upper_right.x >= other._upper_right.x
                and self._upper_right.y >= other._upper_right.y
            ):
                return True
            return False
        else:
            try:
                other = self._VectorClass(other)

                return (
                    self._lower_left.x <= other.x <= self._upper_right.x
                    and self._lower_left.y <= other.y <= self._upper_right.y
                )
            except VectorValueError as error:
                raise TypeError(
                    "other must be a instance of BoundingBox or VectorT."
                ) from error

    def __contains__(self, other: BoundingBoxBase) -> bool:
        return self.contains(other)

    # TODO: allow list of vectors here.
    def extended(
        self: SelfT, other: Union[BoundingBoxBase, VectorT, Iterable[VectorT]]
    ) -> SelfT:
        """
        Return a extended bounding box so that `self` and other are contained

        Args:
            other (BoundingBox): other box
        """

        def _extend_with_vector(bbox, vector) -> None:
            if vector.x < bbox._lower_left.x:
                bbox._lower_left = self._VectorClass(vector.x, bbox._lower_left.y)
            if vector.x > bbox._upper_right.x:
                bbox._upper_right = self._VectorClass(vector.x, bbox._upper_right.y)
            if vector.y < bbox._lower_left.y:
                bbox._lower_left = self._VectorClass(bbox._lower_left.x, vector.y)
            if vector.y > bbox._upper_right.y:
                bbox._upper_right = self._VectorClass(bbox._upper_right.x, vector.y)

        clone = self.clone()

        if isinstance(other, self.__class__):
            if other not in clone:
                if other._lower_left.x < clone._lower_left.x:
                    clone._lower_left = self._VectorClass(
                        other._lower_left.x, clone._lower_left.y
                    )
                if other._lower_left.y < clone._lower_left.y:
                    clone._lower_left = self._VectorClass(
                        clone._lower_left.x, other._lower_left.y
                    )
                if other._upper_right.x > clone._upper_right.x:
                    clone._upper_right = self._VectorClass(
                        other._upper_right.x, clone._upper_right.y
                    )
                if other._upper_right.y > clone._upper_right.y:
                    clone._upper_right = self._VectorClass(
                        clone._upper_right.x, other._upper_right.y
                    )
        else:
            try:
                other = self._VectorClass(other)
            except VectorValueError:
                # other is maybe an Iterable[VectorT]
                try:
                    for p in other:
                        _extend_with_vector(clone, self._VectorClass(p))
                except (TypeError, VectorValueError) as error:
                    raise TypeError(
                        "other must be SelfT, VectorT or Iterable[VectorT]."
                    ) from error
            else:
                _extend_with_vector(clone, other)

        return clone
