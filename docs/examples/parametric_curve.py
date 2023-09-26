import numpy as np

from fibomat.shapes import ParametricCurve
from fibomat import Sample, Q_, U_

# sympy example

import sympy
from sympy.abc import t
from sympy import sin, cos

curve = sympy.Curve(
    [2 * cos(2 * t) + 3 * cos(t), 2 * sin(2 * t) - 3 * sin(t)],
    (t, 0, 2*np.pi)
)

parametric_curve = ParametricCurve.from_sympy_curve(curve, try_length_integration=True)
spline_from_sympy = parametric_curve.to_arc_spline(epsilon=.1)


# manual example

def f(u):
    u = np.asarray(u)

    return np.array(
        (u, u**2)
    ).T


def df(u):
    u = np.asarray(u)

    return np.array(
        (np.full_like(u, 1), 2*u)
    ).T


def d2f(u):
    u = np.asarray(u)

    return np.array(
        (np.full_like(u, 1),  np.full_like(u, 2))
    ).T


def curvature(u):
    u = np.asarray(u)

    return 2 / (1 + 4 * u**2)**1.5


def length_impl(u):
    return u*np.sqrt(4*u**2 + 1)/2 + np.arcsinh(2*u)/4


def length(u_0, u_1):
    return length_impl(u_1) - length_impl(u_0)


# bounding box is not required for now.
# if curvature and length ar not given, these are calculated automatically. This may be inefficient.
parametric_curve = ParametricCurve(
    f, df, d2f,
    domain=(0, 1), bounding_box=None, curvature=curvature, length=length
)
spline_manual = parametric_curve.to_arc_spline()

# plot it

s = Sample()
s.add_annotation(spline_from_sympy.scaled(.5) * U_('µm'))
s.add_annotation(spline_manual.translated((4, 0)) * U_('µm'))
s.plot(rasterize_pitch=Q_('0.001 µm'))
