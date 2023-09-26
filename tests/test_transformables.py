import pytest
from pytest_mock import MockerFixture

from fibomat.linalg import Vector, VectorLike
from fibomat.linalg.transformables import (
    Transformable,
    translate,
    rotate,
    scale,
    mirror,
)
from fibomat.linalg.boundingboxes import BoundingBox
from fibomat.linalg.transformables.transformation_builder import _TransformationBuilder

from stubs import MocKTransformable


class Dummy:
    pass


class FakeTrafo(_TransformationBuilder):
    pass


def _test_scale_rot_without(mocker, meth, meth_name):
    transformable = MocKTransformable()

    meth_spy = mocker.spy(transformable, meth_name)
    translate_spy = mocker.spy(transformable, "_impl_translate")

    meth(transformable, 2)

    meth_spy.assert_called_once_with(2)


def _test_scale_rot_with_vector(mocker, meth, meth_name):
    transformable = MocKTransformable()

    meth_spy = mocker.spy(transformable, meth_name)
    translate_spy = mocker.spy(transformable, "_impl_translate")

    meth(transformable, 2, (3, 4))

    meth_spy.assert_called_once_with(2)
    translate_spy.assert_has_calls(
        [mocker.call(Vector(-3.0, -4.0)), mocker.call(Vector(3, 4))]
    )


def _test_scale_rot_with_center(mocker, meth, meth_name):
    mock_center = mocker.patch.object(
        MocKTransformable, "center", new_callable=mocker.PropertyMock
    )
    mock_center.return_value = Vector(7, 8)

    transformable = MocKTransformable()

    meth_spy = mocker.spy(transformable, meth_name)
    translate_spy = mocker.spy(transformable, "_impl_translate")

    meth(transformable, 2, "center")

    meth_spy.assert_called_once_with(2)
    translate_spy.assert_has_calls(
        [mocker.call(-Vector(7, 8)), mocker.call(Vector(7, 8))]
    )
    mock_center.assert_called_once_with()


def _test_scale_rot_with_pivot(mocker, meth, meth_name):
    mock_pivot = mocker.patch.object(
        MocKTransformable, "pivot", new_callable=mocker.PropertyMock
    )
    mock_pivot.return_value = Vector(7, 8)

    transformable = MocKTransformable()

    meth_spy = mocker.spy(transformable, meth_name)
    translate_spy = mocker.spy(transformable, "_impl_translate")

    meth(transformable, 2, "pivot")

    meth_spy.assert_called_once_with(2)
    translate_spy.assert_has_calls(
        [mocker.call(-Vector(7, 8)), mocker.call(Vector(7, 8))]
    )
    mock_pivot.assert_called_once_with()


def _test_scale_rot_with_unknown(mocker, meth, meth_name):
    transformable = MocKTransformable()

    with pytest.raises(ValueError):
        meth(transformable, 2, "foobarbaz")


class TestTransformables:
    def test_init(self):
        transformable = MocKTransformable(description="foo")
        assert transformable.description == "foo"

    def test_pivot(self):
        transformable = MocKTransformable()

        assert transformable.pivot == transformable.center

        transformable.pivot = lambda self: Vector(-1, -2)
        assert transformable.pivot == Vector(-1, -2)

    def test_translated(self, mocker: MockerFixture):
        transformable = MocKTransformable()

        translated_spy = mocker.spy(transformable, "_impl_translate")
        transformable.translated((1, 2))
        translated_spy.assert_called_once_with((1, 2))

    def test_mirrored(self, mocker: MockerFixture):
        transformable = MocKTransformable()

        translated_spy = mocker.spy(transformable, "_impl_mirror")
        transformable.mirrored((1, 2))
        translated_spy.assert_called_once_with((1, 2))

    def test_rotated_without_origin(self, mocker: MockerFixture):
        _test_scale_rot_without(mocker, MocKTransformable.rotated, "_impl_rotate")

    def test_rotated_with_vector_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_vector(mocker, MocKTransformable.rotated, "_impl_rotate")

    def test_rotated_with_center_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_center(mocker, MocKTransformable.rotated, "_impl_rotate")

    def test_rotated_with_pivot_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_pivot(mocker, MocKTransformable.rotated, "_impl_rotate")

    def test_rotated_with_unknown_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_unknown(mocker, MocKTransformable.rotated, "_impl_rotate")

    def test_scaled_without_origin(self, mocker: MockerFixture):
        _test_scale_rot_without(mocker, MocKTransformable.scaled, "_impl_scale")

    def test_scaled_with_vector_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_vector(mocker, MocKTransformable.scaled, "_impl_scale")

    def test_scaled_with_center_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_center(mocker, MocKTransformable.scaled, "_impl_scale")

    def test_scaled_with_pivot_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_pivot(mocker, MocKTransformable.scaled, "_impl_scale")

    def test_scaled_with_unknown_origin(self, mocker: MockerFixture):
        _test_scale_rot_with_unknown(mocker, MocKTransformable.scaled, "_impl_scale")

    def test_transform(self, mocker: MockerFixture):
        transformable = MocKTransformable()

        calls = []

        translate_spy = mocker.spy(transformable, "_impl_translate")

        def translate_side_effect(*args, **kwargs):
            calls.append("translate")
            return mocker.DEFAULT

        translate_spy.side_effect = translate_side_effect

        rotate_spy = mocker.spy(transformable, "_impl_rotate")

        def rotate_side_effect(*args, **kwargs):
            calls.append("rotate")
            return mocker.DEFAULT

        rotate_spy.side_effect = rotate_side_effect

        scale_spy = mocker.spy(transformable, "_impl_scale")

        def scale_side_effect(*args, **kwargs):
            calls.append("scale")
            return mocker.DEFAULT

        scale_spy.side_effect = scale_side_effect

        mirror_spy = mocker.spy(transformable, "_impl_mirror")

        def mirror_side_effect(*args, **kwargs):
            calls.append("mirror")
            return mocker.DEFAULT

        mirror_spy.side_effect = mirror_side_effect

        # rotate_spy = mocker.spy(transformable, '_impl_translate')
        # scale_spy = mocker.spy(transformable, '_impl_translate')

        transformable.transformed(
            translate((1, 2)) | rotate(2) | scale(3) | mirror((3, 4))
        )

        assert len(calls) == 4

        assert calls[0] == "translate"
        translate_spy.assert_called_once_with((1, 2))

        assert calls[1] == "rotate"
        rotate_spy.assert_called_once_with(2)

        assert calls[2] == "scale"
        scale_spy.assert_called_once_with(3)

        assert calls[3] == "mirror"
        mirror_spy.assert_called_once_with((3, 4))

        with pytest.raises(TypeError):
            transformable.transformed(FakeTrafo())

    def test_trafo_builder(self):
        with pytest.raises(TypeError):
            _TransformationBuilder() | Dummy()

    def test_translate_to(self, mocker):
        transformable = MocKTransformable()

        translate_spy = mocker.spy(transformable, "_impl_translate")

        new_pos = (1, 1)
        transformable.translated_to(new_pos)

        translate_spy.assert_called_once_with(new_pos - transformable.pivot)
