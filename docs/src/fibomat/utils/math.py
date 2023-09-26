"""Provide math helper functions :func:`mod_2pi` and :func:`float_gcd`."""
# pylint: disable=invalid-name
import numpy as np


def mod_2pi(val: float) -> float:
    """Calculate val % 2*pi.

    If val*k % 2*pi == 0, 2*pi is returned for k != 0.

    Args:
        val: value

    Returns:
        float
    """
    mod = np.mod(val, 2*np.pi)
    if np.isclose(mod, 0.):
        if np.isclose(val, 0.):
            return 0.
        return 2*np.pi
    return mod


# https://stackoverflow.com/a/45325587
def float_gcd(a: float, b: float, rtol: float = 1e-05, atol: float = 1e-08):
    """Calculate gcd for float values.

    Args:
        a: first value
        b: second value
        rtol: relative tolerance
        atol: absolut tolerance

    Returns:
        float
    """
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a
