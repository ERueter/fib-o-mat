"""Provide the :func:`self_intersections` and  :func:`curve_intersections` function."""

from typing import Dict, List, Any

from fibomat.shapes import ArcSpline
from fibomat import _libfibomat


def self_intersections(curve: ArcSpline) -> Dict[str, List[Dict[str, Any]]]:
    """Self intersections of curve.

    Args:
        curve (ArcSpline): curve

    Returns:
        Dict[str, List[Dict[str, Any]]]:
            dict with key 'intersections'. 'intersections' contains list where each element is a dict with keys 'seg_1',
            'seg_2', and 'pos' where the former two elements contain the indices of the segments where the intersection
            occurs and the latter the position of the intersection.
    """
    return {
        'intersections': [
            {'seg_1': s1, 'seg_2': s2, 'pos': pos} for s1, s2, pos in
            _libfibomat.self_intersections(curve.arc_spline_impl)
        ]
    }


def curve_intersections(curve_1: ArcSpline, curve_2: ArcSpline) -> Dict[str, List[Dict[str, Any]]]:
    """Intersections between curves.

    Args:
        curve_1 (ArcSpline): first curve
        curve_2 (ArcSpline): second curve

    .. todo:: what is seg_1, seg_2 in coincidences?

    Returns:
        Dict[str, List[Dict[str, Any]]]:
            dict with key 'intersections' and 'coincidences'. 'intersections' contains list where each element is a dict
            with keys 'seg_1', 'seg_2', and 'pos' where the former two elements contain the indices of the segments
            where the intersection occurs and the latter the position of the intersection. 'coincidences' contains a
            list where each element is a dict with keys 'seg_1', 'seg_2', 'start_pos' and 'end_pos'. 'start_pos' and
            'end_pos' indicate the range where the two curves lie on each other.
    """
    intersection, coincidences = _libfibomat.curve_intersections(curve_1.arc_spline_impl, curve_2.arc_spline_impl)

    return {
        'intersections': [
            {'seg_1': s1, 'seg_2': s2, 'pos': pos} for s1, s2, pos in intersection
        ],
        'coincidences': [
            {'seg_1': s1, 'seg_2': s2, 'start_pos': start, 'end_pos': end} for s1, s2, start, end in coincidences
        ]
    }
