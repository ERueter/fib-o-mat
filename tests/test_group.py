import pytest

from fibomat.layout import Group, DimGroup
from fibomat.shapes import Spot


class TestGroup:
    def test_elements(self):
        spots = [Spot((1, 1)), Spot((0, 1))]

        group = Group(spots)

        for spot, group_elem in zip(spots, group.elements):
            assert spot.position == group_elem.position

        with pytest.raises(RuntimeError):
            group.elements[0] = None
