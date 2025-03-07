from typing import Tuple

from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.raster_styles.one_d import Curve
from fibomat.shapes import rasterizedpoints
from fibomat import shapes
from fibomat.units import LengthUnit, TimeUnit, LengthQuantity, TimeQuantity, has_length_dim, scale_to, scale_factor
from fibomat.shapes import Shape, RasterizedPoints, DimShape
from fibomat.mill import Mill
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.curve_tools import fill_with_spiral, rasterize, fill_with_lines
import numpy as np
from typing import Optional
from fibomat.raster_styles.scansequence import ScanSequence


class Spiral(RasterStyle):
    """
    pitch: pitch between points on the spiral
    spiral_pitch: distance between arms of the spiral
    direction: "outwards" for beginning in the center, "inwards" for beginning outside, "out-in" for going out and back in

    """
    def __init__(self, pitch: LengthQuantity, spiral_pitch: LengthQuantity, scan_sequence: ScanSequence, direction: Optional[str]):
        self._pitch = pitch
        self._spiral_pitch = spiral_pitch
        self._scan_sequence = scan_sequence
        if direction:
            if direction not in ["outwards", "inwards", "out-in"]:
                raise ValueError("Direction must be 'outwards', 'inwards' or 'in-out', if not specified it is 'outwards'.")
        self._direction = direction

    @property
    def dimension(self) -> int:
        return 2

    @property
    def pitch(self):
        return self._pitch

    @property
    def scan_sequence(self):
        return self._scan_sequence

    @property
    def direction(self):
        return None

    @property
    def line_style(self):
        return None


    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit,
        rasterize_pitch = 0.1,
        rasterize_epsilon = 0.001
    ) -> RasterizedPattern:
        
        from fibomat import U_
        spiral_pitch = self._spiral_pitch._magnitude

        test = U_(self._spiral_pitch._units)

        scan = self._scan_sequence

        spline_shape = dim_shape.shape.to_arc_spline() 

        filling_spiral = fill_with_spiral(  # spiral filling circumscribed circle of bounding box
            spline_shape, pitch=spiral_pitch
        )

        spiral_spline = filling_spiral.to_arc_spline(rasterize_pitch=rasterize_pitch, epsilon=rasterize_epsilon) # TODO perhaps make parameters variable
        curve = Curve(self._pitch, scan)
        # TODO is out_length_unit correct for the spiral spline?

        rasterized_pat = curve.rasterize(dim_shape=spiral_spline* test, mill=mill, out_length_unit=out_length_unit, out_time_unit=out_time_unit)     
        
        points_inside = [p for p in rasterized_pat._dwell_points if spline_shape.contains(p[0:2])]

        if self._direction == "inwards":
            points_inside.reverse()
        if self._direction == "out-in":
            points_inside = points_inside + list(reversed(points_inside))

        points_inside = np.array(points_inside)

        rasterized_pat._dwell_points = points_inside

        return rasterized_pat
