from typing import Tuple, Union, Optional, TYPE_CHECKING

import numpy as np

from fibomat.shapes.line import Line
from fibomat.shapes.arc import Arc
from fibomat.shapes.biarc import Biarc

if TYPE_CHECKING:
    from fibomat.shapes.parametric_curve import ParametricCurve

from .utils import _make_perp_vector, _Bisector, _PerpBisector, _circle_circle_intersection, _make_fixed_tangent_curvature


# def _make_biarc_segment(parametric_curve: 'ParametricCurve', biarc_domain: Tuple[float, float], epsilon: float):
#     p_0, p_1 = parametric_curve.f(biarc_domain)
#
#     t_0, k_0 = _make_fixed_tangent_curvature(parametric_curve, biarc_domain[0], epsilon, 'left')
#     t_1, k_1 = _make_fixed_tangent_curvature(parametric_curve, biarc_domain[1], epsilon, 'right')
#
#     t_0 /= np.linalg.norm(t_0)
#     t_1 /= np.linalg.norm(t_1)
#
#     t_0_perp = _make_perp_vector(t_0)
#     t_1_perp = _make_perp_vector(t_1)
#
#     b_0 = _PerpBisector(p_0, p_1)
#     b_1 = _PerpBisector(p_0 + t_0, p_1 + t_1)
#
#     if b_0.parallel_to(b_1):
#         return Line(p_0, p_1),
#     else:
#         c_j = b_0.intersect_at(b_1)
#
#         # calc J
#         arc_j_1 = Arc.from_points_center(p_0, p_1, c_j, True)
#         arc_j_2 = Arc.from_points_center(p_0, p_1, c_j, False)
#
#         if arc_j_1.length < arc_j_2.length:
#             j = arc_j_1.midpoint
#             sweep_dir = True
#         else:
#             j = arc_j_2.midpoint
#             sweep_dir = False
#
#         c_0 = _PerpBisector(p_0, j).intersect_at(_Bisector(p_0, p_0 + t_0_perp))
#         arc_0 = Arc.from_points_center_tangent(p_0, j, c_0, unit_tangent_start=t_0)
#
#         c_1 = _PerpBisector(j, p_1).intersect_at(_Bisector(p_1, p_1 + t_1_perp))
#         arc_1 = Arc.from_points_center_tangent(j, p_1, c_1, unit_tangent_end=t_1)
#
#         return arc_0, arc_1


def _make_biarc_from_spiral_segment(
    parametric_curve: 'ParametricCurve', biarc_domain: Tuple[float, float], epsilon: float
):
    # u_0, u_1 = biarc_domain
    p_0, p_1 = parametric_curve.f(biarc_domain)

    t_0, k_0 = _make_fixed_tangent_curvature(parametric_curve, biarc_domain[0], epsilon, 'left')
    t_1, k_1 = _make_fixed_tangent_curvature(parametric_curve, biarc_domain[1], epsilon, 'right')
    t_0 /= np.linalg.norm(t_0)
    t_1 /= np.linalg.norm(t_1)

    # k_0, k_1 = parametric_curve.curvature(biarc_domain)

    b_0 = _PerpBisector(p_0, p_1)
    b_1 = _PerpBisector(p_0 + t_0, p_1 + t_1)

    if b_0.parallel_to(b_1):
        return Line(p_0, p_1),
    else:
        c_j = b_0.intersect_at(b_1)
        r_j = np.linalg.norm(c_j - p_0)

        if np.isclose(k_0, k_1):
            arc = Arc.from_points_center_tangent(start=p_0, end=p_1, center=c_j, unit_tangent_start=t_0)
            return arc.split_at(arc.theta/2)

        if abs(k_0) > abs(k_1):
            r_alpha = abs(1. / k_0)
            p_alpha = p_0
            t_alpha = t_0
            t_alpha_perp = np.sign(k_0) * _make_perp_vector(t_alpha)
            c_alpha = p_alpha + r_alpha * t_alpha_perp
        else:
            r_alpha = abs(1. / k_1)
            p_alpha = p_1
            t_alpha = t_1
            t_alpha_perp = np.sign(k_1) * _make_perp_vector(t_alpha)
            c_alpha = p_alpha + r_alpha * t_alpha_perp

        intersections_j_alpha = _circle_circle_intersection((c_j, r_j), (c_alpha, r_alpha))

        only_other_arc = False

        if len(intersections_j_alpha) == 0:
            # points lie on a circle !?
            # this should not happen if curvatures are not identical of start end endpoint
            raise RuntimeError
            # p_mean = p_0 + p_1
            # j = p_mean / np.linalg.norm(p_mean) * r_j
        elif len(intersections_j_alpha) == 2:
            if np.allclose(intersections_j_alpha[0], p_alpha):
                j = intersections_j_alpha[1]
            elif np.allclose(intersections_j_alpha[1], p_alpha):
                j = intersections_j_alpha[0]
            else:
                raise RuntimeError
        else:
            if not np.allclose(intersections_j_alpha[0], p_alpha):
                raise RuntimeError
            j = intersections_j_alpha[0]
            only_other_arc = True

        if abs(k_0) > abs(k_1):
            p_beta = p_1
            t_beta = t_1
            t_beta_perp = _make_perp_vector(t_beta)
            c_beta = _Bisector(j, c_alpha).intersect_at(_Bisector(p_beta, p_beta + t_beta_perp))

            arc_beta = Arc.from_points_center_tangent(j, p_beta, c_beta, unit_tangent_end=t_beta)

            if only_other_arc:
                return arc_beta,

            arc_alpha = Arc.from_points_center_tangent(p_alpha, j, c_alpha, unit_tangent_start=t_alpha)

        else:
            p_beta = p_0
            t_beta = t_0
            t_beta_perp = _make_perp_vector(t_beta)
            c_beta = _Bisector(j, c_alpha).intersect_at(_Bisector(p_beta, p_beta + t_beta_perp)) # plus or minus?

            arc_beta = Arc.from_points_center_tangent(p_beta, j, c_beta, unit_tangent_start=t_beta)

            if only_other_arc:
                return arc_beta,

            arc_alpha = Arc.from_points_center_tangent(j, p_alpha, c_alpha, unit_tangent_end=t_alpha)

        if abs(k_0) > abs(k_1):
            return arc_alpha, arc_beta
        else:
            return arc_beta, arc_alpha


def _fitting_error(
    param_curve: 'ParametricCurve',
    seg: Union[Tuple[Arc, Arc], Line],
    rasterized_curve_points: np.ndarray
):
    def closest_point(points_, single):
        delta = points_ - single
        dist_ = np.einsum('ij,ij->i', delta, delta)
        return np.argmin(dist_)

    # i_start = np.searchsorted(rasterized_curve_points[:, 0], biarc_domain[0], side='left')
    # i_end = np.searchsorted(rasterized_curve_points[:, 0], biarc_domain[1], side='right')

    if len(seg) == 2:
        biarc = seg

        if len(rasterized_curve_points) == 2:
            # biarcs interpolate points
            return 0.

        j_arcs = biarc[0].end
        # normal = _Bisector(j_arcs, biarc[0].center)
        # res = optimize.root(lambda u: normal(u[0]) - param_curve.f(u[1]), np.array([0., (biarc_domain[1]-biarc_domain[0])/2]))
        # that's a crude approximation to the above lines
        i_j = closest_point(rasterized_curve_points[:, 1:], j_arcs)

        if i_j == 0:
            # biarc[1][i_j] == start of biarc[0] => shift i_j to next point
            i_j += 1

        points_arc_1 = rasterized_curve_points[:i_j, 1:] - biarc[0].center
        dist_arc_1 = np.abs(np.sqrt(np.einsum('ij,ij->i', points_arc_1, points_arc_1)) - biarc[0].radius)

        points_arc_2 = rasterized_curve_points[i_j:, 1:] - biarc[1].center
        dist_arc_2 = np.abs(np.sqrt(np.einsum('ij,ij->i', points_arc_2, points_arc_2)) - biarc[1].radius)

        from fibomat import Sample, U_
        from fibomat.shapes import Spot

        # s = Sample()
        # s.add_annotation(biarc[0] * U_('µm'), color='green')
        # s.add_annotation(biarc[1] * U_('µm'), color='red')
        #
        # for p in points_arc_1:
        #     s.add_annotation(Spot(p + biarc[0].center) * U_('µm'), color='green')
        #
        # for p in points_arc_2:
        #     s.add_annotation(Spot(p + biarc[1].center) * U_('µm'), color='red')
        #
        # # for p in param_curve.rasterize_at(0.01):
        # #     s.add_annotation(Spot(p) * U_('µm'))
        #
        # s.plot()

        if len(dist_arc_1) == 0:
            return np.max(dist_arc_2)
        elif len(dist_arc_2) == 0:
            return np.max(dist_arc_1)
        else:
            return max(np.max(dist_arc_1), np.max(dist_arc_2))
    elif len(seg) == 1:
        line_: Line = seg[0]

        points = rasterized_curve_points[:, 1:]

        # https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line
        s = np.asarray(line_.end - line_.start)
        v = points

        proj = np.outer(np.dot(v, s) / np.dot(s, s), s)

        delta = proj - v
        dist = np.einsum('ij,ij->i', delta, delta)

        return np.sqrt(np.max(dist))

    else:
        raise RuntimeError


def _approximate_param_curve_greedy(
    param_func: 'ParametricCurve',
    parameter_values: np.ndarray,
    epsilon: float
):
    # u_min, u_max = parameter_values[0], parameter_values[-1]
    # u_0, u_1 = u_min, u_min
    #
    # h = (u_max - u_min) / 2

    i_min = 0
    i_max = len(parameter_values) - 1

    # rasterized_curve_points = param_func.f(u_rasterized_int)
    rasterized_curve_points = np.empty(shape=(len(parameter_values), 3), dtype=float)
    rasterized_curve_points[:, 0] = parameter_values
    rasterized_curve_points[:, 1:] = param_func.f(parameter_values)

    from fibomat import Sample, U_
    from fibomat.shapes import Spot
    s = Sample()
    for point in rasterized_curve_points:
        s.add_annotation(Spot(point[1:]) * U_('µm'))

    segments = []

    end_reached = False
    testing = False
    step_size_reduced = False

    prev_segment = None
    prev_h: Optional[float] = None

    i_seg_start, i_seg_end = i_min, i_max
    h = (i_seg_end - i_seg_start) // 2 + 2

    if h == 0:
        raise RuntimeError

    while True:
        i_seg_end = i_seg_start + h

        if i_seg_end > i_max:
            h = i_max - i_seg_start
            # u_1 = u_max
            end_reached = True

            interval = (rasterized_curve_points[i_seg_start, 0], rasterized_curve_points[i_max, 0])
        else:
            interval = (rasterized_curve_points[i_seg_start, 0], rasterized_curve_points[i_seg_end, 0])

        # new_segment = _make_biarc_from_spiral_segment(param_func, interval)

        if h <= 2:
            p_0, p_1 = param_func.f(interval)

            t_0, _ = _make_fixed_tangent_curvature(param_func, interval[0], epsilon, 'left')
            t_1, _ = _make_fixed_tangent_curvature(param_func, interval[1], epsilon, 'right')

            new_segment = Biarc(p_0, p_1, t_0, t_1).segments
            error = 0
        else:

            # t_0, t_1 = param_func.df(interval)
            # if np.allclose(t_0, 0) or np.allclose(t_1, 0):
            #     print('dsfdsf')
            #     new_segment = _make_biarc_segment(param_func, interval, epsilon)
            # else:
            #     new_segment = _make_biarc_from_spiral_segment(param_func, interval, epsilon)
            new_segment = _make_biarc_from_spiral_segment(param_func, interval, epsilon)

            error = _fitting_error(param_func, new_segment, rasterized_curve_points[i_seg_start:i_seg_end+1])

        # print('domain', interval, i_seg_start, i_seg_end)
        # print(h, error)

        if error < epsilon:

            if end_reached:
                segments.extend(new_segment)
                break

            if testing and step_size_reduced:
                # def plot():
                #     for seg in prev_segment:
                #         s.add_annotation(seg * U_('µm'))
                #     s.plot(rasterize_pitch=Q_('0.001 µm'))
                #
                # plot()

                segments.extend(prev_segment)
                i_seg_start = i_seg_end - prev_h

                testing = False
                step_size_reduced = False
            else:
                # save current state and try again in next round with extended interval
                prev_segment = new_segment
                prev_h = h
                h *= 2

                testing = True
        else:
            end_reached = False
            if testing:
                # save state from last round
                # print('int =', (u_0, u_1 - h_temp), 'saving', 'error check')

                # def plot():
                #     for seg in prev_segment:
                #         s.add_annotation(seg * U_('µm'))
                #     s.plot(rasterize_pitch=Q_('0.001 µm'))
                #
                # plot()

                segments.extend(prev_segment)
                i_seg_start = i_seg_end - prev_h

                testing = False
            else:
                # reduce step size
                if h > 10:
                    h = h // 2
                else:
                    h -= 1

                # if h <= 3:
                #     # raise RuntimeError
                #     print('interesting')

                step_size_reduced = True

                # T0D0: Find a good criterion for aborting
                # if h < 0.01 * epsilon:
                #     raise RuntimeError('Fitting does not converge.')

    # return biarcs

    return segments
