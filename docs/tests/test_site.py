import pytest

import numpy as np

from fibomat.site import Site
from fibomat.linalg import Vector, DimVector
from fibomat.units import U_
from fibomat.shapes import Spot
from fibomat.pattern import Pattern


class TestSite:
    def test_init(self):
        center = Vector(1, 2) * U_('µm')
        fov = Vector(3, 4) * U_('µm')

        site = Site(center, fov, description='foo')

        assert site.description == 'foo'
        assert site.center == center
        assert site.fov == fov

    def test_fov_bbox(self):
        site_1 = Site(DimVector(), Vector(3, 4) * U_('µm'), description='foo')
        assert site_1.fov == Vector(3, 4) * U_('µm')
        assert site_1.square_fov == Vector(4, 4) * U_('µm')

        with pytest.raises(RuntimeError):
            site_1.bounding_box

        site_2 = Site(DimVector(), Vector(4, 3) * U_('µm'), description='foo')
        assert site_2.square_fov == Vector(4, 4) * U_('µm')

        # dynamic fov
        site_3 = Site(DimVector())
        site_3.create_pattern(Spot((-1, 0)) * U_('µm'), None, None)
        site_3.create_pattern(Spot((2, 2)) * U_('µm'), None, None)

        assert site_3.fov == Vector(3, 2) * U_('µm')
        bbox_3 = site_3.bounding_box
        assert bbox_3.width.m == 3
        assert bbox_3.height.m == 2
        assert bbox_3.center.vector == (.5, 1)

        # fixed fov
        site_4 = Site(DimVector(), Vector(3, 4) * U_('µm'))
        # add a shape outside of site
        site_4.create_pattern(Spot((5, 5)) * U_('µm'), None, None)
        assert site_4.fov == Vector(3, 4) * U_('µm')

        bbox_4 = site_4.bounding_box
        assert bbox_4.width.m == 0
        assert bbox_4.height.m == 0
        assert bbox_4.center.vector == (5, 5)

    def test_add_pattern(self):
        site = Site(DimVector())

        assert site.empty

        pattern_1 = Pattern(Spot((1, 1)) * U_('µm'), None, None)
        site.add_pattern(pattern_1)

        pattern_2 = Pattern(Spot((2, 2)) * U_('µm'), None, None)
        site += pattern_2

        site_3 = site.create_pattern(Spot((3, 3)) * U_('µm'), None, None)
        assert site_3

        assert not site.empty

        assert len(site.patterns) == 3
        assert site.patterns[0].center.vector == (1, 1)
        assert site.patterns[1].center.vector == (2, 2)
        assert site.patterns[2].center.vector == (3, 3)

    # TODO: add tests
    def test_add_layout(self):
        pass

    def test_transformations(self):
        site = Site(DimVector(), (3, 6) * U_('µm'))
        site += Pattern(Spot((1, 1)) * U_('µm'), None, None)

        assert site.patterns_absolute[0].center == (1, 1) * U_('µm')

        site = site.translated((1, 1) * U_('µm'))
        assert site.center == (1, 1) * U_('µm')
        # pattern must not be translates because these are relative to site's center
        assert site.patterns[0].center == (1, 1) * U_('µm')
        # but the absolute position must have changed
        assert site.patterns_absolute[0].center == (2, 2) * U_('µm')

        site = site.rotated(np.pi / 2)
        # fov  should be flipped
        assert site.fov == (6, 3) * U_('µm')
        assert site.center == (-1, 1) * U_('µm')

        assert site.patterns[0].center == (-1, 1) * U_('µm')
        assert site.patterns_absolute[0].center == (-2, 2) * U_('µm')

        site = site.rotated(np.pi/2, origin='center')
        assert site.fov == (3, 6) * U_('µm')
        assert site.center == (-1, 1) * U_('µm')
        assert site.patterns[0].center == (-1, -1) * U_('µm')
        assert site.patterns_absolute[0].center == (-2, 0) * U_('µm')

        site = site.rotated(np.pi)
        # no change of fov here.
        assert site.fov == (3, 6) * U_('µm')

        # reset site (with other center!)
        site = Site((1, 1) * U_('µm'), (3, 6) * U_('µm'))
        site += Pattern(Spot((1, 1)) * U_('µm'), None, None)

        site = site.scaled(2)

        assert site.fov == (6, 12) * U_('µm')
        assert site.center == (2, 2) * U_('µm')
        assert site.patterns[0].center == (2, 2) * U_('µm')
        assert site.patterns_absolute[0].center == (4, 4) * U_('µm')

        # reset site (with other center!)
        site = Site((1, 0) * U_('µm'), (3, 6) * U_('µm'))
        site += Pattern(Spot((1, 1)) * U_('µm'), None, None)

        site = site.mirrored((1, 1) * U_('µm'))

        assert site.fov == (6, 3) * U_('µm')
        assert site.center == (0, 1) * U_('µm')
        assert site.patterns[0].center == (1, 1) * U_('µm')

        # TODO: Think about mirroring and finish test cases

        # assert site.patterns_absolute[0].center == ???

        # reset site (with other center!)
        site = Site((1, 0) * U_('µm'), (3, 6) * U_('µm'))
        site += Pattern(Spot((1, 1)) * U_('µm'), None, None)

        site = site.mirrored((1, 0) * U_('µm'))

        assert site.fov == (3, 6) * U_('µm')
        # TODO: Think about mirroring and finish test cases

        # test possible rotation angles / mirror axes
        with pytest.raises(ValueError):
            site.rotated(1.234)

        with pytest.raises(ValueError, match='Sites can only'):
            site.mirrored((2, 3) * U_('µm'))



