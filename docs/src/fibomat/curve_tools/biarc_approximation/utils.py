from __future__ import annotations
from typing import Tuple

import numpy as np


def _make_fixed_tangent_curvature(
    parametric_curve, param: float, epsilon: float, side: str
) -> Tuple[np.ndarray, float]:
    t = parametric_curve.df(param)
    k = parametric_curve.curvature(param)

    if np.allclose(t, 0):
        if side == 'left':
            t = parametric_curve.df(param + epsilon)
            k = parametric_curve.curvature(param + epsilon)
        elif side == 'right':
            t = parametric_curve.df(param - epsilon)
            k = parametric_curve.curvature(param - epsilon)
        else:
            raise ValueError(f'unknown side {side}')

        assert not np.allclose(t, 0)

    return t, k


def _make_perp_vector(p):
    return np.array([-p[1], p[0]], dtype=p.dtype if isinstance(p, np.ndarray) else float)


class _LinearFunction:
    def __init__(self, p, p_sup):
        self.p = np.array(p)
        self.p_sup = np.array(p_sup)

    def __repr__(self):
        return 't |-> {} + t * {}'.format(self.p_sup, self.p)

    def __call__(self, u):
        return self.p_sup + u*self.p

    def parallel_to(self, other: _LinearFunction):
        return np.isclose(
            np.dot(
                self.p / np.linalg.norm(self.p),
                other.p / np.linalg.norm(other.p)
            ), 1.
        )

    def intersect_at(self, other: _LinearFunction):
        m = np.array([self.p, -other.p]).T
        b = other.p_sup - self.p_sup

        res = np.linalg.solve(m, b)

        return self.p_sup + res[0] * self.p


class _PerpBisector(_LinearFunction):
    def __init__(self, p_0, p_1):
        super().__init__(_make_perp_vector(p_1 - p_0), (p_0 + p_1) / 2)


class _Bisector(_LinearFunction):
    def __init__(self, p_0, p_1):
        p_0 = np.array(p_0)
        p_1 = np.array(p_1)
        super().__init__(p_1 - p_0, (p_0 + p_1) / 2)


def _circle_circle_intersection(circ_1, circ_2):
    c_0, r_0 = circ_1
    c_1, r_1 = circ_2

    c_0 = np.array(c_0)
    c_1 = np.array(c_1)

    d = np.linalg.norm(c_0 - c_1)

    if d > r_0 + r_1:
        raise RuntimeError
    elif d < abs(r_0 - r_1):
        raise RuntimeError
    elif np.isclose(d, 0.) and np.isclose(r_0, r_1):
        return tuple()
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
            return p_1,
        else:
            return p_1, p_2
