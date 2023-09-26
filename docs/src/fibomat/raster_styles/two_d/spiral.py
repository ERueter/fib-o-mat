from typing import Tuple

from fibomat.rasterizing.styles import areastyle
from fibomat.shapes import rasterizedpoints
from fibomat import shapes
from fibomat import units


class Spiral(areastyle.AreaStyle):
    def __init__(self, pitch: units.LengthQuantity, spiral_pitch: units.LengthQuantity):
        self._pitch = pitch
        self._spiral_pitch = spiral_pitch

    def rasterize(
            self,
            dim_shape: Tuple[shapes.Shape, units.LengthUnit],
            repeats: int,
            time_unit: units.TimeQuantity,
            length_unit: units.LengthUnit
    ) -> rasterizedpoints.RasterizedPoints:
        pass

    def explode(self, *args):
        pass
