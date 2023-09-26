from typing import Optional, Union, List

import numpy as np

from fibomat.linalg import VectorLike, Vector, Transformable
from fibomat.linalg.transformables.transformable_base import VectorT, BBoxT
from fibomat.shapes.arc import Arc, ArcSpline, ArcSplineCompatible
from fibomat.shapes import Line


class Biarc(Transformable, ArcSplineCompatible):
    """
    https://www.ryanjuckett.com/biarc-interpolation/
    """

    @classmethod
    def _make_biarc_seg(cls, start: Vector, end: Vector, tangent: Vector, reversed: bool) -> Union[Arc, Line, None]:
        """Create a arc or line from ``start`` to ``end`` and tangent ``tangent`` at start.

        Args:
            start (Vector): start
            end (Vector):
            tangent (Vector): tangent at start
            reversed (bool): if True, the arc is reversed

        Returns:
            Union[Arc, Line, None]: None is returned if start == end, Line is returned if (end - start) || tangent and
                                    Arc otherwise
        """
        # https://www.ryanjuckett.com/biarc-interpolation/
        tangent = tangent.normalized()
        normal = Vector(-tangent.y, tangent.x)
        end_to_start = end - start

        if np.allclose(end_to_start, 0.):
            # raise cls._ZeroArcAngle('arc angle ist zero.')
            return None

        u_denominator = 2 * normal.dot(end_to_start)

        if np.isclose(u_denominator, 0.):
            if reversed:
                return Line(end, start)
            else:
                return Line(start, end)

        u = end_to_start.dot(end_to_start) / u_denominator

        center = start + u * normal
        radius = abs(u)

        center_to_start = (start - center) / radius
        center_to_end = (end - center) / radius

        sweep = center_to_start.cross(center_to_end)

        delta_angle = np.sign(sweep) * np.arccos(center_to_start.dot(center_to_end))

        start_angle = (start - center).angle_about_x_axis
        end_angle = start_angle + delta_angle

        if reversed:
            return Arc(
                radius=radius,
                start_angle=end_angle, end_angle=start_angle,
                sweep_dir=not(sweep > 0),
                center=center
            )
        else:
            return Arc(
                radius=radius,
                start_angle=start_angle, end_angle=end_angle,
                sweep_dir=sweep > 0,
                center=center
            )

    @classmethod
    def _make_biarc_segs(
        cls, p_1: Vector, p_2: Vector, t_1: Vector, t_2: Vector, d: float
    ) -> List[Union[Arc, Line, None]]:
        """Create arcs of biarc for given d value. See source for definition of ``d``.

        Args:
            p_1 (Vector): start of biarc
            p_2 (Vector): end of biarc
            t_1 (Vector): tangent at start
            t_2 (Vector): tangent at end
            d (float): d value

        Returns:
            Tuple[Arc, Arc]
        """
        midpoint = (p_1 + p_2 + d * (t_1 - t_2)) / 2

        seg_1 = cls._make_biarc_seg(p_1, midpoint, t_1, False)
        seg_2 = cls._make_biarc_seg(p_2, midpoint, t_2, True)

        return [seg for seg in [seg_1, seg_2] if seg]

    def __init__(
        self,
        p_1: VectorLike, p_2: VectorLike,
        t_1: VectorLike, t_2: VectorLike,
        description: Optional[str] = None
    ):
        """Interpolate a biarc to given points and tangents.

        For interpolation details see references.

        References:
            - https://www.ryanjuckett.com/biarc-interpolation/
            - https://ieeexplore.ieee.org/document/5390085

        Args:
            p_1 (Vector): start of biarc
            p_2 (Vector): end of biarc
            t_1 (Vector): tangent at start
            t_2 (Vector): tangent at end
            description (str, optional): optional description
        """
        super().__init__(description)

        p_1 = Vector(p_1)
        p_2 = Vector(p_2)

        t_1 = Vector(t_1).normalized()
        t_2 = Vector(t_2).normalized()

        t = t_1 + t_2
        v = p_2 - p_1

        if np.allclose(v, 0.):
            raise ValueError('p_1 == p_2. No biarc can be interpolated.')

        d_denominator = 2. * (1. - t_1.dot(t_2))

        if np.isclose(d_denominator, 0.):
            if np.isclose(v.dot(t_2), 0.):
                # create to semicircles
                midpoint = p_1 + v / 2
                c_1 = p_1 + .25 * v
                c_2 = p_1 + .75 * v

                sweep = v.cross(t_2)

                segments = [
                    Arc.from_points_center(p_1, midpoint, c_1, sweep < 0),
                    Arc.from_points_center(midpoint, p_2, c_2, sweep > 0)
                ]
            else:
                d = v.dot(v) / (4. * v.dot(t_2))
                segments = self._make_biarc_segs(p_1, p_2, t_1, t_2, d)
        else:
            d_discriminant = v.dot(t) * v.dot(t) + d_denominator * v.dot(v)
            d = (-v.dot(t) + np.sqrt(d_discriminant)) / d_denominator

            segments = self._make_biarc_segs(p_1, p_2, t_1, t_2, d)

        assert 1 <= len(segments) <= 2
        self._biarc = ArcSpline.from_segments(segments)

    @property
    def segments(self) -> List[Union[Arc, Line]]:
        return self._biarc.segments

    def to_arc_spline(self) -> ArcSpline:
        return self._biarc

    @property
    def center(self) -> VectorT:
        """Center is mean value of start of first segment and end of last segment (hence, it is compatible with ArSpline)

        Returns:

        """
        return self._biarc.center

    @property
    def bounding_box(self) -> BBoxT:
        return self._biarc.bounding_box

    def _impl_translate(self, trans_vec: VectorT) -> None:
        self._biarc._impl_translate(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        self._biarc._impl_rotate(theta)

    def _impl_scale(self, fac: float) -> None:
        self._biarc._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: VectorT) -> None:
        self._biarc._impl_mirror(mirror_axis)

# from fibomat import Sample, U_
# from fibomat.shapes import Spot
#
# s = Sample()
#
# # two arcs
# # biarc = Biarc(Vector(0, 1), Vector(3, 0), Vector(-1, 0), Vector(0, 1))
#
# # only one arc
# # biarc = Biarc(Vector(-1, 0), Vector(1, 0), Vector(1, -1), Vector(-1, -1))
#
# # line + arc
# # biarc = Biarc(Vector(0, 0), Vector(3, -1), Vector(1, 0), Vector(0, -1))
#
# # line + line
# # biarc = Biarc(Vector(0, 0), Vector(3, 0), Vector(1, 0), Vector(1, 0))
#
# # two arcs (degenerate case, two semicircles)
# # biarc = Biarc(Vector(0, 1), Vector(0, -1), Vector(1, 0), Vector(1, 0))
#
# # two arcs (degenerate case, but no semicircles)
# biarc = Biarc(Vector(0, 1), Vector(1, -1), Vector(1, 0), Vector(1, 0))
#
#
# # print()
# #
# # arc = arc_center_radius_from_start_end_tangent(Vector(100, 100), Vector(200, 0), Vector(-1, 0))
# # print(arc)
#
#
# # s.add_annotation(arc * U_('µm'))
#
# for a in biarc.segments:
#     print(a.start, a.end)
#     s.add_annotation(a * U_('µm'))
#
# # s.add_annotation(Arc.from(Vector(100, 100), Vector(200, 0), Vector(-1, 0)) * U_('µm'))
#
# s.plot()
