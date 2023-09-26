import itertools

import pytest

import numpy as np

from fibomat.linalg.boundingboxes import BoundingBox, DimBoundingBox
from fibomat.units import U_


class TestBoundingBoxes:
    def test_init(self):
        def _check_init(class_, lower_left, upper_right):
            box = class_(lower_left, upper_right)

            assert np.allclose(box._lower_left, lower_left)
            assert np.allclose(box._upper_right, upper_right)

        _check_init(BoundingBox, (-1, -1), (1, 1))
        _check_init(BoundingBox, (1, 1), (1, 1))

        with pytest.raises(ValueError):
            BoundingBox((1, 1), (-1, -1))

        _check_init(DimBoundingBox, (-1, -1) * U_('µm'), (1, 1) * U_('µm'))
        _check_init(DimBoundingBox, (1, 1) * U_('µm'), (1, 1) * U_('µm'))
        # dim_bbox = DimBoundingBox((-1, -1) * U_('µm'), (1, 1) * U_('µm'))
        # dim_bbox = DimBoundingBox((1, 1) * U_('µm'), (1, 1) * U_('µm'))

        with pytest.raises(ValueError):
            DimBoundingBox((1, 1) * U_('µm'), (-1, -1) * U_('µm'))

    def test_from_points(self):
        with pytest.raises(ValueError):
            BoundingBox.from_points([])

        with pytest.raises(ValueError):
            BoundingBox.from_points([(1, 2, 3), (1, 2)])

        with pytest.raises(ValueError):
            BoundingBox.from_points([(1, 2, 3), (1, 2, 3)])

        points = np.random.random((10, 2))
        box = BoundingBox.from_points(points)
        assert box._lower_left.x == np.min(points[:, 0])
        assert box._lower_left.y == np.min(points[:, 1])
        assert box._upper_right.x == np.max(points[:, 0])
        assert box._upper_right.y == np.max(points[:, 1])

        with pytest.raises(ValueError):
            DimBoundingBox.from_points([])

        with pytest.raises(ValueError):
            DimBoundingBox.from_points([(1, 2, 3) * U_('µm'), (1, 2) * U_('µm')])

        with pytest.raises(ValueError):
            DimBoundingBox.from_points([(1, 2, 3) * U_('µm'), (1, 2, 3) * U_('µm')])

        points = np.random.random((10, 2))
        dim_points = [p * U_('µm') for p in points]
        box = DimBoundingBox.from_points(dim_points)
        assert box._lower_left.x == np.min(points[:, 0] * U_('µm'))
        assert box._lower_left.y == np.min(points[:, 1] * U_('µm'))
        assert box._upper_right.x == np.max(points[:, 0] * U_('µm'))
        assert box._upper_right.y == np.max(points[:, 1] * U_('µm'))

    def test_eq_close_to(self):
        box = BoundingBox((0, 0), (1, 1))
        dim_box = DimBoundingBox((0, 0) * U_('µm'), (1, 1) * U_('µm'))

        assert box == box
        assert box.close_to(box)
        assert dim_box == dim_box
        assert dim_box.close_to(dim_box)

        assert box != BoundingBox((0, 0.1), (1, 1))
        assert not box.close_to(BoundingBox((0, 0.1), (1, 1)))
        assert dim_box != DimBoundingBox((0, 0.1) * U_('µm'), (1, 1) * U_('µm'))
        assert not dim_box.close_to(DimBoundingBox((0, 0.1) * U_('µm'), (1, 1) * U_('µm')))

    def test_lower_left_upper_right(self):
        box = BoundingBox((0, 0), (1, 2))
        dim_box = DimBoundingBox((0, 0) * U_('µm'), (1, 2) * U_('µm'))

        assert box.lower_left.close_to((0, 0))
        assert box.upper_right.close_to((1, 2))

        assert dim_box.lower_left.close_to((0, 0) * U_('µm'))
        assert dim_box.upper_right.close_to((1, 2) * U_('µm'))

    def test_width_height(self):
        box = BoundingBox((0, 0), (1, 2))
        dim_box = DimBoundingBox((0, 0) * U_('µm'), (1, 2) * U_('µm'))

        assert np.isclose(box.width, 1)
        assert np.isclose(box.height, 2)

        assert np.isclose(dim_box.width, 1 * U_('µm'))
        assert np.isclose(dim_box.height, 2 * U_('µm'))

    def test_center(self):
        box = BoundingBox((0, 0), (1, 2))
        dim_box = DimBoundingBox((0, 0) * U_('µm'), (1, 2) * U_('µm'))

        assert np.allclose(box.center, (.5, 1))
        assert np.allclose(dim_box.center, (.5, 1) * U_('µm'))

    def test_clone(self):
        box = BoundingBox((0, 0), (1, 2))
        dim_box = DimBoundingBox((0, 0) * U_('µm'), (1, 2) * U_('µm'))

        assert id(box) != id(box.clone())
        assert id(dim_box) != id(dim_box.clone())

    def test_scaled(self):
        def _test_scaled(box_, scale):
            scaled_box = box_.scaled(scale)
            assert scaled_box.center.close_to(box_.center)
            assert np.isclose(scaled_box.width, box_.width * scale)
            assert np.isclose(scaled_box.height, box_.height * scale)

        box = BoundingBox((.5, 1.), (1.5, 2.))
        dim_box = DimBoundingBox((.5, 1.) * U_('µm'), (1.5, 2.) * U_('µm'))

        _test_scaled(box, 2.)
        _test_scaled(dim_box, 2.)

    def test_overlaps_with_contains(self):

        def _test_overlaps_with_contains(bbox, other_width_2, mult):
            for i_x in np.linspace(-3, 3, 5):
                i_x = i_x * mult
                for i_y in np.linspace(-3, 3, 5):
                    i_y = i_y * mult
                    other = bbox.__class__((i_x-other_width_2, i_y-other_width_2), (i_x+other_width_2, i_y+other_width_2))

                    if abs(i_x) > 2*mult or abs(i_y) > 2*mult:
                        assert not bbox.overlaps_with(other)
                    else:
                        assert bbox.overlaps_with(other)

                    if abs(i_x) < 2*mult and abs(i_y) < 2*mult:
                        assert bbox.contains(other)
                        assert other in bbox
                    else:
                        assert not bbox.contains(other)
                        assert other not in bbox

        _test_overlaps_with_contains(BoundingBox((-2, -2), (2, 2)), .5 / 2, 1.)
        _test_overlaps_with_contains(DimBoundingBox((-2, -2) * U_('µm'), (2, 2) * U_('µm')), .5 / 2 * U_('µm'), U_('µm'))

    def test_contains_vector(self):
        def _test_contains(bbox, mult):
            for i_x in np.linspace(-3, 3, 3):
                i_x = i_x * mult
                for i_y in np.linspace(-3, 3, 3):
                    i_y = i_y * mult

                    vec = bbox._VectorClass(i_x, i_y)

                    if abs(i_x) > 2*mult or abs(i_y) > 2*mult:
                        assert not bbox.contains(vec)
                    else:
                        assert bbox.contains(vec)

        _test_contains(BoundingBox((-2, -2), (2, 2)), 1.)
        _test_contains(DimBoundingBox((-2, -2) * U_('µm'), (2, 2) * U_('µm')), U_('µm'))

        with pytest.raises(TypeError):
            BoundingBox((-2, -2), (2, 2)).contains((1, 2, 3))

        with pytest.raises(TypeError):
            BoundingBox((-2, -2), (2, 2)).contains('foo')

    def test_extended(self):
        def _test_extend_vector(bbox, mult):
            for i_x in np.linspace(-3, 3, 5):
                i_x = i_x * mult
                for i_y in np.linspace(-3, 3, 5):
                    i_y = i_y * mult
                    vec = bbox._VectorClass(i_x, i_y)

                    extended = bbox.extended(vec)

                    assert extended.contains(vec)

                    if abs(i_x) > 2*mult or abs(i_y) > 2*mult:
                        assert bbox != extended
                    else:
                        assert bbox == extended

        def _test_extend(bbox, other_width_2, mult):
            for i_x in np.linspace(-3, 3, 5):
                i_x = i_x * mult
                for i_y in np.linspace(-3, 3, 5):
                    i_y = i_y * mult
                    other = bbox.__class__((i_x-other_width_2, i_y-other_width_2), (i_x+other_width_2, i_y+other_width_2))

                    extended = bbox.extended(other)
                    assert extended.contains(bbox)
                    assert extended.contains(other)

        _test_extend_vector(BoundingBox((-2, -2), (2, 2)), 1.)
        _test_extend_vector(DimBoundingBox((-2, -2) * U_('µm'), (2, 2) * U_('µm')), U_('µm'))

        points = list(itertools.combinations_with_replacement(np.linspace(-3, 3, 5), 2))
        dim_points = [p * U_('µm') for p in points]

        bbox = BoundingBox((-2, -2), (2, 2)).extended(points)
        assert bbox.center.close_to((0, 0))
        assert bbox.width == 6.
        assert bbox.height == 6.

        dim_bbox = DimBoundingBox((-2, -2) * U_('µm'), (2, 2) * U_('µm')).extended(dim_points)
        assert dim_bbox.center.close_to((0, 0) * U_('µm'))
        assert dim_bbox.width == 6. * U_('µm')
        assert dim_bbox.height == 6. * U_('µm')

        with pytest.raises(TypeError):
            BoundingBox((-2, -2), (2, 2)).extended([1, 2, 3])

        _test_extend(BoundingBox((-2, -2), (2, 2)), .5 / 2, 1.)
        _test_extend(DimBoundingBox((-2, -2) * U_('µm'), (2, 2) * U_('µm')), .5 / 2 * U_('µm'), U_('µm'))

    def test_repr(self):
        str(BoundingBox((-2, -2), (2, 2)))
