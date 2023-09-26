from fibomat.linalg import Vector
from fibomat.pattern import Pattern
from fibomat.shapes.spot import Spot
from fibomat.units import U_
from fibomat.mill.mill import MillBase
from fibomat.default_backends import StubRasterStyle

# from stubs import MockDimShape

from stubs import MockDimShape


class TestPattern:
    def test_props(self):
        dim_shape = Spot((0, 0)) * U_("nm")
        mill = MillBase()
        raster_style = StubRasterStyle(1)
        kwargs = {"a": 1, "b": 2}
        description = "foo"

        pattern = Pattern(
            dim_shape=dim_shape,
            mill=mill,
            raster_style=raster_style,
            description=description,
            **kwargs
        )

        assert pattern.dim_shape == dim_shape
        assert pattern.mill == mill
        assert pattern.raster_style == raster_style
        assert pattern.kwargs == kwargs
        assert pattern.description == description

    def test_transformations(self, mocker):
        mock_shape = MockDimShape()

        translate_spy = mocker.spy(mock_shape, "_impl_translate")
        rotate_spy = mocker.spy(mock_shape, "_impl_rotate")
        scale_spy = mocker.spy(mock_shape, "_impl_scale")
        mirror_spy = mocker.spy(mock_shape, "_impl_mirror")

        pattern = Pattern(dim_shape=mock_shape, mill=None, raster_style=None)

        pattern._impl_translate(Vector() * U_("m"))
        translate_spy.assert_called_once()

        pattern._impl_rotate(1)
        rotate_spy.assert_called_once()

        pattern._impl_scale(2)
        scale_spy.assert_called_once()

        pattern._impl_mirror(Vector(1, 0) * U_("m"))
        mirror_spy.assert_called_once()

        assert pattern.center == mock_shape.center
        assert pattern.bounding_box == mock_shape.bounding_box
