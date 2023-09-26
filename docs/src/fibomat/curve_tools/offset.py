"""Provide the offsetting functionality."""
from typing import List, Optional

from fibomat import _libfibomat
from fibomat.shapes import ArcSpline


def offset(curve: ArcSpline, delta: float, direction: Optional[str] = None) -> List[ArcSpline]:
    """Offset a list of curves. The offset direction depends on the orientation of the curves and the sign of delta.

    ::

                         | delta > 0 | delta < 0 |
        -----------------+-----------+-----------+
        pos. orientation | inwards   | outwards  |
        -----------------+-----------+-----------+
        neg. orientation | outwards  | inwards   |
        -----------------+-----------+-----------+

    Args:
        curve (ArcSpline): curve to be offsetted.
        delta (float): offset distance
        direction (str, optional): offset direction. Only needed if curve is closed. Should be `inwards` or `outwards`.

    Returns:
        List[ArcSplines]

    Raises:
        ValueError: Raised if direction is not provided an curve is closed.
    """
    # if delta <= 0.:
    #     raise ValueError("delta <= 0.")

    raw_curve = curve.arc_spline_impl

    if curve.is_closed:

        if not direction:
            raise ValueError('direction must be provided if curve is closed.')

        if direction == 'outwards':
            if curve.orientation:
                delta *= -1
        else:
            if not curve.orientation:
                delta *= -1

    res_raw_curves = _libfibomat.offset_curve(raw_curve, delta)

    return [ArcSpline(raw_curve) for raw_curve in res_raw_curves]


def offset_with_islands(islands: List[ArcSpline], delta: float, outer_curve: Optional[ArcSpline] = None):
    """Offset a region with islands. The offset direction will always be outwards.

    Args:
        islands (List[ArcSpline]): list of islands. All islands must be closed curves and orientation must be True.
        delta (float): offset distance
        outer_curve (ArcSpline, optional): optional outer curve, which will be offsetted inwards.

    Returns:
        Dict[str, Any]:
            dict with keys 'islands' and 'outer_curve'. 'islands' contains a list of new islands and 'outer_curve'
            contains a new outer curve if outer_curve was not None.
    """
    # if delta <= 0.:
    #     raise ValueError("delta <= 0.")

    raw_islands = [island.arc_spline_impl for island in islands]
    raw_outer_curve = outer_curve.arc_spline_impl if outer_curve else None

    res_raw_curves = _libfibomat.offset_with_islands(raw_islands, raw_outer_curve, delta)

    islands = [ArcSpline(raw_curve) for raw_curve in res_raw_curves[0]]
    outer_curve = [ArcSpline(raw_curve) for raw_curve in res_raw_curves[1]]

    return {'islands': islands, 'outer_curve': outer_curve}


SAFETY: int = 1000
"""max steps for deflate"""


def deflate(
    arc_spline: ArcSpline, pitch: float, *, n_steps: Optional[int] = None, distance: Optional[float] = None
):
    """Deflate a given arc spline (completely).

    The original curve is not included.

    Args:
        arc_spline (ArcSpline): curve to be inflated
        pitch (float): offset delta.
        n_steps (int, optional): if provided at maximum n_steps will be performed.
        distance (float, optional): if provided, a total distance 'distance' will be deflated.

    Returns:
        List[ArcSpline]

    Raises:
        ValueError: Raised if steps < 0.
        ValueError: Raised if distance < 0.
        ValueError: Raised if steps and distance are provided.
        RuntimeError: Raised if more than :attr:`SAFETY` steps are performed.
    """
    if n_steps and not distance:
        if n_steps < 0:
            raise ValueError('steps must be greater than 0.')
        max_steps = n_steps
    elif not n_steps and distance:
        if distance < 0.:
            raise ValueError('distance must be greater than 0.')
        max_steps = int(distance / pitch)
    else:
        max_steps = SAFETY
        if n_steps and distance:
            raise ValueError('distance and steps cannot be set both.')

    deflated_curves = []

    deflate_stack = [arc_spline]
    new_deflate_stack = []

    finished = False

    steps = 0

    while not finished and steps < max_steps:
        for to_be_inflated in deflate_stack:
            newly_deflated = offset(to_be_inflated, pitch, 'inwards')
            if newly_deflated:
                deflated_curves.extend(newly_deflated)
                new_deflate_stack.extend(newly_deflated)

        deflate_stack = new_deflate_stack
        new_deflate_stack = []

        steps += 1
        if steps == SAFETY:
            raise RuntimeError('steps > SAFETY. If you know you need so many steps, increase SAFETY.')

        if not deflate_stack:
            finished = True

    return deflated_curves


def inflate(
    arc_spline: ArcSpline, pitch: float, *, n_steps: Optional[int] = None, distance: Optional[float] = None
):
    """Inflate a given arc spline (completely).

    The original curve is not included.

    Note, any one of n_steps and distance must be provided.

    Args:
        arc_spline (ArcSpline): curve to be inflated
        pitch (float): offset delta.
        n_steps (int, optional): if provided at maximum n_steps will be performed.
        distance (float, optional): if provided, a total distance 'distance' will be deflated.

    Returns:
        List[ArcSpline]

    Raises:
        ValueError: Raised if steps < 0.
        ValueError: Raised if distance < 0.
        ValueError: Raised if none of steps and distance are provided.
    """
    if n_steps and not distance:
        if n_steps < 0:
            raise ValueError('steps must be greater than 0.')
        max_steps = n_steps
    elif not n_steps and distance:
        if distance < 0.:
            raise ValueError('distance must be greater than 0.')
        max_steps = int(distance / pitch)
    else:
        raise ValueError('Any of distance or steps must be set.')

    inflated_curves = []
    inflate_stack = [arc_spline]
    new_inflate_stack = []

    for _ in range(max_steps):
        for to_be_inflated in inflate_stack:
            newly_deflated = offset(to_be_inflated, pitch, 'outwards')
            if newly_deflated:
                inflated_curves.extend(newly_deflated)
                new_inflate_stack.extend(newly_deflated)

        inflate_stack = new_inflate_stack
        new_inflate_stack = []

    return inflated_curves
