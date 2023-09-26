"""Provide the :func:`combine_curves` function."""
from typing import Dict, List

from fibomat.shapes import ArcSpline
from fibomat import _libfibomat


def combine_curves(curve_1: ArcSpline, curve_2: ArcSpline, mode: str) -> Dict[str, List[ArcSpline]]:
    """Combine two curves. The combining method is given by `mode`.

    `mode` can be:

        - `union`
        - `xor`
        - `exclude`
        - `intersect`

    See Usage guide for further details.

    Args:
        curve_1 (ArcSpline): first curve
        curve_2 (ArcSpline): second curve
        mode (str): combining mode

    Returns:
        List[ArcSpline]: list of new ArcSpline
    """
    remaining, subtracted = _libfibomat.combine_curves(curve_1.arc_spline_impl, curve_2.arc_spline_impl, mode)

    return {
        'remaining': [ArcSpline(raw) for raw in remaining],
        'subtracted': [ArcSpline(raw) for raw in subtracted]
    }


# def trim_curves(curve_1: shapes.Curve, curve_2: shapes.Curve) -> Dict[str, List[shapes.Curve]]:
#     """Trim curve `curve_2` on curve `curve_1`.
#
#     .. warning:: no working currently
#
#     Args:
#         curve_1 (shapes.Curve): trimming geometry, curve must be closed.
#         curve_2 (shapes.Curve): curve to be trimmed
#
#     Returns:
#         Dict[str, List[shapes.Curve]]: dict with keys `inner` and `outer`. Curves in former category lie in the inside
#                                        of `curve_1` and curves in latter category on the outside.
#     """
#
#     trimmed_curves = _libfibomat.trim_curves(curve_1._curve, curve_2._curve)
#     return {
#         'inside': [shapes.Curve.from_raw(raw) for raw in trimmed_curves[0]],
#         'outside': [shapes.Curve.from_raw(raw) for raw in trimmed_curves[1]]
#     }
