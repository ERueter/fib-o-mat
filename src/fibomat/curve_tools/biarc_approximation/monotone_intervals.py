from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING

import enum

import numpy as np

if TYPE_CHECKING:
    from fibomat.shapes.parametric_curve import ParametricCurve

from .utils import _make_perp_vector


@enum.unique
class _CircleRelations(enum.Enum):
    CIRCLE_1_IN_CIRCLE_2 = enum.auto()
    CIRCLE_2_IN_CIRCLE_1 = enum.auto()
    IDENTICAL = enum.auto()
    NO_INTERSECTION = enum.auto()
    OVERLAP = enum.auto()


@enum.unique
class _CurvatureState(enum.Enum):
    INCREASING = enum.auto()
    DECREASING = enum.auto()
    CONSTANT = enum.auto()
    INFINITE = enum.auto()

    UNDEFINED = enum.auto()

    # at begin of new segment
    NONE = enum.auto()


@enum.unique
class _FlushMode(enum.Enum):
    NO_FLUSH = enum.auto()
    FLUSH_WITH_TRANSITION = enum.auto()
    FLUSH_WITHOUT_TRANSITION = enum.auto()


@enum.unique
class IntervalType(enum.Enum):
    LINE = enum.auto()
    ARC = enum.auto()
    BIARC = enum.auto()
    SPIRAL = enum.auto()


def _osculating_circle(param_curve: 'ParametricCurve', parameter: float):
    curvature = param_curve.curvature(parameter)
    radius = abs(1 / curvature)

    tangent = param_curve.df(parameter)
    tangent /= np.linalg.norm(tangent)

    t_perp = np.sign(curvature) * _make_perp_vector(tangent)
    center_osc_circle = param_curve.f(parameter) + radius * t_perp

    return center_osc_circle, radius


def _circle_relation(circle_1: Tuple[np.ndarray, float], circle_2: Tuple[np.ndarray, float]):
    c_1, r_1 = circle_1
    c_2, r_2 = circle_2

    d = np.linalg.norm(c_1 - c_2)

    if np.allclose(c_1, c_2) and np.allclose(r_1, r_2):
        return _CircleRelations.IDENTICAL
    if r_1 >= d + r_2:
        return _CircleRelations.CIRCLE_2_IN_CIRCLE_1
    elif r_2 >= d + r_1:
        return _CircleRelations.CIRCLE_1_IN_CIRCLE_2
    elif d >= r_1 + r_2 + d:
        return _CircleRelations.NO_INTERSECTION
    else:
        return _CircleRelations.OVERLAP


def interval_type(n_points: int, curvature_state: _CurvatureState):
    if n_points < 2:
        raise RuntimeError
    elif n_points == 2:
        return IntervalType.BIARC
    else:
        if curvature_state == _CurvatureState.INFINITE:
            return IntervalType.LINE
        elif curvature_state == _CurvatureState.CONSTANT:
            return IntervalType.ARC
        elif curvature_state in (_CurvatureState.INCREASING, _CurvatureState.DECREASING):
            return IntervalType.SPIRAL
        else:
            raise RuntimeError


def intervals_by_osculating_circles(
    param_curve: ParametricCurve, parameter_values: np.ndarray, epsilon: float
) -> List[Tuple[IntervalType, np.ndarray]]:

    # TODO: INFINITE should be really ZERO_CURVATURE

    if len(parameter_values) == 0:
        raise RuntimeError

    prev_osculating_circle = None
    curvature_state = _CurvatureState.NONE

    intervals = []
    current_interval = []

    for i, param in enumerate(parameter_values):
        if np.allclose(param_curve.df(param), 0):
            if i == len(parameter_values) - 1:
                k = param_curve.curvature(param - epsilon)
            else:
                k = param_curve.curvature(param + epsilon)
        else:
            k = param_curve.curvature(param)

        flush = _FlushMode.NO_FLUSH

        if np.isclose(k, 0.):
            if curvature_state == _CurvatureState.NONE:
                current_interval.append(param)
                curvature_state = _CurvatureState.UNDEFINED
            elif curvature_state == _CurvatureState.UNDEFINED:
                current_interval.append(param)
                curvature_state = _CurvatureState.INFINITE
            elif curvature_state == _CurvatureState.INFINITE:
                current_interval.append(param)
            elif curvature_state == _CurvatureState.DECREASING:
                # finish current interval and start a new one but without a connecting biarc segment
                flush = _FlushMode.FLUSH_WITHOUT_TRANSITION
            else:
                flush = _FlushMode.FLUSH_WITH_TRANSITION
        else:
            osculating_circle = _osculating_circle(param_curve, param)

            if curvature_state == _CurvatureState.NONE:
                current_interval.append(param)
                curvature_state = _CurvatureState.UNDEFINED
            elif curvature_state == _CurvatureState.INFINITE:
                # this could be handled better most likely
                flush = _FlushMode.FLUSH_WITH_TRANSITION
            elif curvature_state == _CurvatureState.CONSTANT:
                relation = _circle_relation(osculating_circle,  prev_osculating_circle)

                if relation == _CircleRelations.IDENTICAL:
                    current_interval.append(param)
                else:
                    flush = _FlushMode.FLUSH_WITH_TRANSITION
            elif curvature_state == _CurvatureState.UNDEFINED and prev_osculating_circle is None:
                # Can this happen at all?
                flush = _FlushMode.FLUSH_WITHOUT_TRANSITION
            elif curvature_state == _CurvatureState.UNDEFINED:
                current_interval.append(param)

                # determine curvature state
                relation = _circle_relation(osculating_circle,  prev_osculating_circle)
                if relation == _CircleRelations.NO_INTERSECTION:
                    raise RuntimeError(
                        'No intersection between osculating circles. Try to decrease the rasterization pitch.'
                    )
                elif relation == _CircleRelations.IDENTICAL:
                    current_interval.append(param)
                    curvature_state = _CurvatureState.CONSTANT
                    # if np.isclose(osculating_circle[1], prev_osculating_circle[1]):
                    #     curvature_state = _CurvatureState.CONSTANT
                    # else:
                    #     flush = _FlushMode.FLUSH_WITHOUT_TRANSITION
                else:
                    if relation == _CircleRelations.CIRCLE_1_IN_CIRCLE_2:
                        curvature_state = _CurvatureState.INCREASING
                    else:
                        curvature_state = _CurvatureState.DECREASING
            elif curvature_state in (_CurvatureState.INCREASING, _CurvatureState.DECREASING):
                relation = _circle_relation(osculating_circle,  prev_osculating_circle)

                if relation == _CircleRelations.NO_INTERSECTION:
                    raise RuntimeError(
                        'No intersection between osculating circles. Try to decrease the rasterization pitch.'
                    )
                elif curvature_state == _CurvatureState.INCREASING and relation == _CircleRelations.CIRCLE_1_IN_CIRCLE_2:
                    current_interval.append(param)
                    prev_osculating_circle = osculating_circle
                elif curvature_state == _CurvatureState.DECREASING and relation == _CircleRelations.CIRCLE_2_IN_CIRCLE_1:
                    current_interval.append(param)
                    prev_osculating_circle = osculating_circle
                else:  # _CircleRelations.OVERLAP or something else => make a transition segment and new interval
                    flush = _FlushMode.FLUSH_WITH_TRANSITION
            else:
                raise RuntimeError('Failed.')

            prev_osculating_circle = osculating_circle

        if flush == _FlushMode.FLUSH_WITH_TRANSITION:
            prev_osculating_circle = None

            intervals.append((interval_type(len(current_interval), curvature_state), np.array(current_interval)))
            intervals.append((IntervalType.BIARC, np.array([parameter_values[i-1], param])))

            curvature_state = _CurvatureState.NONE
            prev_osculating_circle = None

            current_interval = [param]
        elif flush == _FlushMode.FLUSH_WITHOUT_TRANSITION:
            prev_osculating_circle = None

            current_interval.append(param)
            intervals.append((interval_type(len(current_interval), curvature_state), np.array(current_interval)))

            curvature_state = _CurvatureState.NONE
            prev_osculating_circle = None

            current_interval = [param]

    if flush == _FlushMode.NO_FLUSH:
        intervals.append((interval_type(len(current_interval), curvature_state), np.array(current_interval)))

    return intervals
