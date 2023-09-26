from __future__ import annotations
from typing import Optional

from fibomat import shapes
from fibomat import linalg
from fibomat.linalg import boundingbox


class Ring(shapes.Shape):
    def __init__(self, inner_r: float, outer_r: float, center: Optional[linalg.VectorLike] = None):
        super().__init__()

        self._inner_r = float(inner_r)
        self._outer_r = float(outer_r)

        if self._inner_r >= self._outer_r:
            raise RuntimeError

        self._center = linalg.Vector(center) if center is not None else linalg.Vector(0, 0)

    @property
    def inner_r(self) -> float:
        return self._inner_r

    @property
    def outer_r(self) -> float:
        return self._outer_r

    def __repr__(self) -> str:
        return '{}(inner_r={!r}, outer_r={!r} center={!r})'.format(
            self.__class__.__name__, self._inner_r, self._outer_r, self._center
        )

    @property
    def bounding_box(self) -> boundingbox.BoundingBox:
        return boundingbox.BoundingBox(
            self._center-(self._outer_r, self._outer_r), self._center+(self._outer_r, self._outer_r)
        )

    @property
    def is_closed(self) -> bool:
        return True

    @property
    def center(self) -> linalg.Vector:
        return self._center.clone()

    def translate(self, trans_vec: linalg.VectorLike) -> Ring:
        self._center += linalg.Vector(trans_vec)
        return self

    def simple_rotate(self, theta: float) -> None:
        pass

    def simple_scale(self, s: float) -> None:
        s = float(s)
        self._inner_r *= s
        self._outer_r *= s


# plot an example

from fibomat import sample
from fibomat import units
from fibomat import pattern

ring = Ring(inner_r=1, outer_r=2)

ring_sample = sample.Sample(f'{ring}')

site = ring_sample.create_site(([0, 0], units.U_('µm')), dim_fov=([5, 5], units.U_('µm')))

site += pattern.Pattern(dim_shape=(ring, units.U_('µm')), mill=None, raster_style=None)

ring_sample.plot(show=True, fullscreen=False)
