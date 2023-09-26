from typing import Union, Tuple

import numpy as np

from fibomat.shapes.line import Line
from fibomat.shapes.arc import Arc
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.curve_tools.intersections import curve_intersections
from fibomat.curve_tools.offset import offset_with_islands, offset
from fibomat.linalg import angle_between, Vector
from fibomat.linalg.helpers import GeomLine
from fibomat.utils.math import mod_2pi



class NonSmoothableError(RuntimeError):
    """"""


def make_tangent_vector(start: np.ndarray, end: np.ndarray):
    tangent = np.asarray(end - start, dtype=float)
    tangent /= np.linalg.norm(tangent)

    return tangent


def make_normal_vector(other: np.ndarray):
    normal = np.array((-other[1], other[0]))

    return normal


def intersection_on_arc(arc: Arc, intersection: np.ndarray):
    intersection_angle = Vector(intersection - arc.center).angle_about_x_axis

    norm_intersection_angle = mod_2pi(intersection_angle - arc.start_angle)
    norm_end_arc_angle = mod_2pi(arc.end_angle - arc.start_angle)

    if not arc.sweep_dir:
        return 0. <= norm_end_arc_angle <= norm_intersection_angle
    else:
        return 0. <= norm_intersection_angle <= norm_end_arc_angle


def intersection_on_line(line: Line, intersection: np.ndarray):
    geom_line = GeomLine(direction=line.end-line.start, support=line.start)

    return 0. <= geom_line.find_param(intersection) <= 1.


def arc_arc_intersection(arc_0: Arc, arc_1: Arc):
    c_0, r_0 = arc_0.center, arc_0.radius
    c_1, r_1 = arc_1.center, arc_1.radius

    c_0 = np.array(c_0)
    c_1 = np.array(c_1)

    d = np.linalg.norm(c_0 - c_1)

    if d > r_0 + r_1:
        raise RuntimeError('No intersection (d > r_0 + r_1)')
    elif d < abs(r_0 - r_1):
        raise RuntimeError('No intersection (d < abs(r_0 - r_1))')
    elif np.isclose(d, 0.) and np.isclose(r_0, r_1):
        raise RuntimeError('No intersection (arcs are identical)')
    else:
        a = (r_0**2 - r_1**2 + d**2) / (2 * d)

        h = np.sqrt(r_0**2 - a**2)

        c = c_0 + a * (c_1 - c_0) / d

        p_1 = np.array([
            c[0] + h * (c_1[1] - c_0[1]) / d,
            c[1] - h * (c_1[0] - c_0[0]) / d
        ])

        p_2 = np.array([
            c[0] - h * (c_1[1] - c_0[1]) / d,
            c[1] + h * (c_1[0] - c_0[0]) / d
        ])

        if np.allclose(p_1, p_2):
            intersection = p_1
        else:
            if intersection_on_arc(arc_0, p_1) and intersection_on_arc(arc_1, p_1):
                intersection = p_1
            elif intersection_on_arc(arc_0, p_2) and intersection_on_arc(arc_1, p_2):
                intersection = p_2
            else:
                raise NonSmoothableError

        return intersection


# http://paulbourke.net/geometry/circlesphere/
def arc_line_intersection(arc: Arc, line: Line):
    x_1 = line.start.x
    x_2 = line.end.x
    x_3 = arc.center.x

    y_1 = line.start.y
    y_2 = line.end.y
    y_3 = arc.center.y

    r = arc.radius

    a = (x_2 - x_1)**2 + (y_2 - y_1)**2
    b = 2 * ((x_2 - x_1) * (x_1 - x_3) + (y_2 - y_1) * (y_1 - y_3))
    c = x_3**2 + y_3**2 + x_1**2 + y_1**2 - 2 * (x_3*x_1 + y_3*y_1) - r**2

    discr = b**2 - 4 * a * c

    if discr < 0:
        raise RuntimeError
    elif np.isclose(discr, 0):
        raise RuntimeError
    else:
        u_1 = (-b + np.sqrt(discr)) / (2*a)
        u_2 = (-b - np.sqrt(discr)) / (2*a)

        p_1 = line.start + u_1 * (line.end - line.start)
        p_2 = line.start + u_2 * (line.end - line.start)

        if intersection_on_arc(arc, p_1) and intersection_on_line(line, p_1):  #
            intersection = p_1
        elif intersection_on_arc(arc, p_2) and intersection_on_line(line, p_2):  #
            intersection = p_2
        else:
            raise NonSmoothableError

        return intersection


def line_line_intersection(line_1: Line, line_2: Line):
    geom_line_1 = GeomLine.make_bisector(line_1.start, line_1.end)
    geom_line_2 = GeomLine.make_bisector(line_2.start, line_2.end)

    intersection = geom_line_1.intersect_at(geom_line_2)

    if not intersection_on_line(line_1, intersection) or not intersection_on_line(line_2, intersection):
        raise NonSmoothableError

    return geom_line_1.intersect_at(geom_line_2)


def make_arc_func(segment: Arc, other_tangent: np.array, kink: np.ndarray, radius: float):
    phi_0 = segment.start_angle
    arc_dir = -1 if not segment.sweep_dir else 1
    theta = segment.theta
    arc_center = segment.center
    arc_radius = segment.radius

    normal_vec = make_normal_vector(segment.unit_tangent_at(0.25 * segment.theta))

    if np.dot(normal_vec, other_tangent) < 0:
        normal_vec *= -1

    if np.dot(normal_vec, kink - segment.center) > 0:
        arc_normal_dir = 1
    else:
        arc_normal_dir = -1

    # def arc_func(t_):
    #     phi = arc_dir * t_ * theta + phi_0
    #     return (
    #         arc_center
    #         + arc_radius * np.array([np.cos(phi), np.sin(phi)])
    #         - arc_normal_dir * radius * np.array([-np.cos(phi), -np.sin(phi)])
    #     )
    #
    # def arc_param(t_):
    #     phi = arc_dir * t_ * theta + phi_0
    #     return (
    #         arc_center
    #         + arc_radius * np.array([np.cos(phi), np.sin(phi)])
    #     )

    arc_offset_seg = Arc(
        radius=arc_normal_dir * radius + segment.radius,
        start_angle=segment.start_angle,
        end_angle=segment.end_angle,
        sweep_dir=segment.sweep_dir,
        center=segment.center
    )

    # return arc_func, arc_param, arc_offset_seg
    return arc_offset_seg


def make_segments(
    left_vertex: np.ndarray, kink_vertex: np.ndarray, right_vertex: np.ndarray, radius: float
):
    left_bulge = left_vertex[2]
    right_bulge = kink_vertex[2]

    kink = np.asarray(kink_vertex[:2], dtype=float)
    left = np.asarray(left_vertex[:2], dtype=float)
    right = np.asarray(right_vertex[:2], dtype=float)

    # build curve segments. the left segment has an inverted direction
    if np.isclose(left_bulge, 0.):
        left_segment = Line(start=kink, end=left)
        left_tangent_start = make_tangent_vector(start=kink, end=left)
    else:
        left_segment = Arc.from_bulge(start=kink, end=left, bulge=-left_bulge)
        left_tangent_start = left_segment.unit_tangent_start

    if np.isclose(right_bulge, 0.):
        right_segment = Line(start=kink, end=right)
        right_tangent_start = make_tangent_vector(start=kink, end=right)
    else:
        right_segment = Arc.from_bulge(start=kink, end=right, bulge=right_bulge)
        right_tangent_start = right_segment.unit_tangent_start

    left_normal_vec = make_normal_vector(left_tangent_start)
    right_normal_vec = make_normal_vector(right_tangent_start)

    if isinstance(left_segment, Line):
        if np.dot(left_normal_vec, right_tangent_start) < 0:
            left_normal_dir = -1
        else:
            left_normal_dir = 1

        left_offset = Line(
            start=left_segment.start + radius * left_normal_dir * left_normal_vec,
            end=left_segment.end + radius * left_normal_dir * left_normal_vec
        )

    else:
        # left_func, left_param,
        left_offset = make_arc_func(left_segment, right_tangent_start, kink, radius)

    if isinstance(right_segment, Line):
        if np.dot(right_normal_vec, left_tangent_start) < 0:
            right_normal_dir = -1
        else:
            right_normal_dir = 1

        right_offset = Line(
            start=right_segment.start + radius * right_normal_dir * right_normal_vec,
            end=right_segment.end + radius * right_normal_dir * right_normal_vec
        )

    else:
        # right_func, right_param,
        right_offset = make_arc_func(right_segment, left_tangent_start, kink, radius)

    return (left_segment, right_segment), (left_offset, right_offset)


def make_smoothing_arc(
    left_segment: Union[Line, Arc],
    left_offset: Union[Line, Arc],
    right_segment: Union[Line, Arc],
    right_offset: Union[Line, Arc],
):
    if isinstance(left_segment, Arc) and isinstance(right_segment, Arc):
        smoothing_arc_center = arc_arc_intersection(left_offset, right_offset)
        smoothing_arc_start = Vector(smoothing_arc_center - left_segment.center).normalized_to(left_segment.radius) + left_segment.center
        smoothing_arc_end = Vector(smoothing_arc_center - right_segment.center).normalized_to(right_segment.radius) + right_segment.center
    elif isinstance(left_segment, Line) and isinstance(right_segment, Line):
        smoothing_arc_center = line_line_intersection(left_offset, right_offset)

        # https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line
        s1 = left_segment.end - left_segment.start
        v1 = smoothing_arc_center - left_segment.start
        smoothing_arc_start = s1.dot(v1) / s1.dot(s1) * s1 + left_segment.start
        s2 = right_segment.end - right_segment.start
        v2 = smoothing_arc_center - right_segment.start
        smoothing_arc_end = s2.dot(v2) / s2.dot(s2) * s2 + right_segment.start
    elif isinstance(left_segment, Line) and isinstance(right_segment, Arc):
        smoothing_arc_center = arc_line_intersection(arc=right_offset, line=left_offset)
        s1 = left_segment.end - left_segment.start
        v1 = smoothing_arc_center - left_segment.start
        smoothing_arc_start = s1.dot(v1) / s1.dot(s1) * s1 + left_segment.start
        smoothing_arc_end = Vector(smoothing_arc_center - right_segment.center).normalized_to(right_segment.radius) + right_segment.center
    elif isinstance(left_segment, Arc) and isinstance(right_segment, Line):
        smoothing_arc_center = arc_line_intersection(arc=left_offset, line=right_offset)
        smoothing_arc_start = Vector(smoothing_arc_center - left_segment.center).normalized_to(left_segment.radius) + left_segment.center
        s2 = right_segment.end - right_segment.start
        v2 = smoothing_arc_center - right_segment.start
        smoothing_arc_end = s2.dot(v2) / s2.dot(s2) * s2 + right_segment.start
    else:
        raise RuntimeError

    return smoothing_arc_start, smoothing_arc_center, smoothing_arc_end


def make_smoothed_vertices(
    smoothing_arc_points: Tuple,
    left_vertex, kink_vertex, right_vertex,
    left_segment, right_segment
):
    smoothing_arc_start, smoothing_arc_center, smoothing_arc_end = smoothing_arc_points

    if isinstance(left_segment, Line):
        new_left_vertex = left_vertex
    else:
        new_left_vertex = (
            *left_vertex[:2],
            np.tan(
                angle_between(
                    left_segment.center - smoothing_arc_start,
                    left_segment.center - left_vertex[:2]
                ) / 4
            ) * np.sign(left_vertex[2])
        )

    if isinstance(right_segment, Line):
        new_right_vertex = (*smoothing_arc_end, 0)
    else:
        new_right_vertex = (
            *smoothing_arc_end,
            np.tan(
                angle_between(
                    right_segment.center - smoothing_arc_end,
                    right_segment.center - right_vertex[:2]
                ) / 4
            ) * np.sign(kink_vertex[2])
        )

    sweep_matrix = np.ones(shape=(3, 3), dtype=float)
    sweep_matrix[0, 1:] = np.asarray(smoothing_arc_start)
    sweep_matrix[1, 1:] = np.asarray(kink_vertex[:2])
    sweep_matrix[2, 1:] = np.asarray(smoothing_arc_end)

    smooth_arc_sweep_dir = 1 if np.linalg.det(sweep_matrix) > 0 else -1

    new_arc = (
        *smoothing_arc_start,
        np.tan(
            angle_between(
                smoothing_arc_center - smoothing_arc_start,
                smoothing_arc_center - smoothing_arc_end
            ) / 4
        ) * smooth_arc_sweep_dir
    )

    return new_left_vertex, new_arc, new_right_vertex


def smooth(arc_spline: ArcSpline, radius: float):
    kinks = arc_spline.kinks()

    if kinks:
        vertices = list(arc_spline.vertices)

        wrap = lambda index: index % len(vertices)

        index_offset = 0

        for i_kink in kinks:
            i_kink_with_offset = i_kink + index_offset
            # print(i_kink, vertices[i_last_kink:i_kink-1])
            kink_vertex = vertices[wrap(i_kink_with_offset)]

            left_vertex = vertices[wrap(i_kink_with_offset-1)]
            right_vertex = vertices[wrap(i_kink_with_offset+1)]

            (left_segment, right_segment), (left_offset, right_offset) = make_segments(
                left_vertex, kink_vertex, right_vertex, radius
            )

            smoothing_arc_points = make_smoothing_arc(
                left_segment, left_offset, right_segment, right_offset
            )

            new_left_vertex, new_arc, new_right_vertex = make_smoothed_vertices(
                smoothing_arc_points, left_vertex, kink_vertex, right_vertex, left_segment, right_segment
            )

            vertices[i_kink_with_offset-1] = np.asarray(new_left_vertex)
            vertices[i_kink_with_offset] = np.asarray(new_right_vertex)
            vertices[i_kink_with_offset:i_kink_with_offset] = [np.asarray(new_arc)]

            index_offset += 1

        return ArcSpline(np.array(vertices), arc_spline.is_closed)

    # no kinks found, no smoothing must be done
    return arc_spline
