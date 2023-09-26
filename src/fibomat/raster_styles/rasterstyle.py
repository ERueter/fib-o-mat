"""Provide the :class:`RasterStyle` class."""
from __future__ import annotations

import abc
import copy

from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes.dim_shape import DimShape
from fibomat.units import LengthUnit, TimeUnit
from fibomat.mill import Mill


class RasterStyle(abc.ABC):
    """Base class to define a raster style.

    All raster styles have the required parameters as attributes (e.g. pitch, ...).
    Further, all raster styles must implement a rasterize method which applies the raster style to a shape.
    """
    def __repr__(self) -> str:
        return self.__class__.__name__

    def clone(self) -> RasterStyle:
        """Deep copy the raster style.

        Returns:
            RasterStyle
        """
        return copy.deepcopy(self)

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the required dimensionality a shape must have to apply this raster style
        Returns:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:
        """Rasterize the given shape.

        The number of repeats and dwell_time defined in the mill object must be applied!
        The points and dwell times in the returned RasterizedPoints object must be scaled according to out_length_unit
        and out_time_unit.

        Args:
            dim_shape (DimObjLike[Shape, LengthUnit]): shpae with length unit to be rasterized.
            mill (Mill): mill
            out_length_unit (LengthUnit): length unit of returned RasterizedPoints
            out_time_unit (TimeUnit): time unit of returned RasterizedPoints

        Returns:
            RasterizedPoints
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def explode(
    #     self,
    #     dim_shape: DimObjLike[Shape, LengthUnit],
    #     mill: Mill,
    #     out_length_unit: LengthUnit,
    #     out_time_unit: TimeUnit
    # ) -> Sequence[Tuple[Shape, RasterStyle]]:
    #     """
    #     Apply raster style to given shape and break it down to "simpler", lower dimensional operation.
    #     E.g. a Rect shape could be exploded into a lot of individual lines.
    #
    #     If no reduction is possible, the same shape (maybe scaled, according to out_length_unit) and RasterStyle will
    #     be returned.
    #
    #     Args:
    #         dim_shape:
    #         mill:
    #         out_length_unit:
    #         out_time_unit:
    #
    #     Returns:
    #
    #     """
    #     raise NotImplementedError
