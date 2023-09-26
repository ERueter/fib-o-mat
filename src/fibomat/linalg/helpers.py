from __future__ import annotations

import numpy as np

from fibomat.linalg.vectors import VectorLike, Vector


def make_perp_vector(vec: VectorLike) -> Vector:
    vec = Vector(vec)
    return Vector(-vec.y, vec.x)


class GeomLine:
    def __init__(self, direction: VectorLike, support: VectorLike):
        self._direction = np.asarray(Vector(direction))
        self._support = np.asarray(Vector(support))

        if np.allclose(self._direction, 0.):
            raise RuntimeError('direction must not be the null vector.')

    @classmethod
    def make_bisector(cls, p_0: VectorLike, p_1: VectorLike):
        p_0 = Vector(p_0)
        p_1 = Vector(p_1)

        return cls(p_1 - p_0, (p_0 + p_1) / 2)

    @classmethod
    def make_perp_bisector(cls, p_0: VectorLike, p_1: VectorLike):
        p_0 = Vector(p_0)
        p_1 = Vector(p_1)

        return cls(make_perp_vector(p_1 - p_0), (p_0 + p_1) / 2)

    def __call__(self, u):
        return self._support + u * self._direction

    def parallel_to(self, other: GeomLine):
        return np.isclose(
            np.dot(
                self._direction / np.linalg.norm(self._direction),
                other._direction / np.linalg.norm(other._direction)
            ), 1.
        )

    def intersect_at_param(self, other: GeomLine):
        m = np.array([self._direction, -other._direction]).T
        b = other._support - self._support

        res = np.linalg.solve(m, b)

        return res[0]

    def intersect_at(self, other: GeomLine):
        res = self.intersect_at_param(other)

        return self._support + res * self._direction

    def find_param(self, point: VectorLike):
        point = Vector(point)

        if np.isclose(self._direction[0], 0.):
            if not np.isclose(point.x, self._support[0]):
                raise ValueError('point lies not on GeomLine.')
            s = (point.y - self._support[1]) / self._direction[1]
        elif np.isclose(self._direction[1], 0.):
            if not np.isclose(point.y, self._support[1]):
                raise ValueError('point lies not on GeomLine.')
            s = (point.x - self._support[0]) / self._direction[0]
        else:
            s_0 = (point.x - self._support[0]) / self._direction[0]
            s_1 = (point.y - self._support[1]) / self._direction[1]

            if not np.isclose(s_0, s_1):
                print(s_0, s_1)
                raise ValueError('point lies not on GeomLine.')

            s = s_0

        return s
