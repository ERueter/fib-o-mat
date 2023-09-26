"""Provide Curve raster style."""

from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.units import LengthUnit, TimeUnit, LengthQuantity
from fibomat.dimensioned_object import DimObjLike, DimObj
from fibomat.mill import Mill
from fibomat.shapes import Shape


class Linear(RasterStyle):
    def __init__(self, pitch_u: LengthQuantity, pitch_v: LengthQuantity, theta: float = 0.):
        """
        Args:
            pitch_u (LengthQuantity): pitch on lines
            pitch_v (LengthQuantity): pitch between lines
            theta (float): rotation angle of lines
        """
        self._pitch_u = pitch_u
        self._pitch_v = pitch_v
        self._theta = theta

    def __repr__(self) -> str:
        return '{}(pitch_u={!r}, pitch_v={!r}, theta={!r})'.format(
            self.__class__.__name__, self._pitch_u, self._pitch_v, self._theta
        )

    @property
    def dimension(self) -> int:
        return 2

    def rasterize(
        self,
        dim_shape: DimObjLike[Shape, LengthUnit],
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPoints:
        raise NotImplementedError
