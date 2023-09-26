import numpy as np

import pytest

from fibomat.linalg.vectors import Vector, DimVector, VectorValueError, angle_between
from fibomat.units import U_


class Dummy:
    pass


class TestVectors:
    @pytest.mark.parametrize(
        "expected, args, kwargs",
        [
            ((0.0, 0.0), tuple(), {}),
            ((1.0, 2.0), (1.0, 2.0), {}),
            ((1.0, 2.0), ((1.0, 2.0),), {}),
            ((1.0, 2.0), ([1.0, 2.0],), {}),
            pytest.param(
                (1.0, 2.0),
                ([1.0],),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                ([1.0, 2.0, 3.0],),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                (["foo", 2],),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                ([2, "foo"],),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                (Dummy(),),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            ((1.0, 2.0), tuple(), {"x": 1.0, "y": 2.0}),
            pytest.param(
                (1.0, 2.0),
                (1.0,),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                (1.0, 2.0, 3.0),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                (1.0, 2.0, 3.0, 4.0),
                {},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                tuple(),
                {"x": 1.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                tuple(),
                {"y": 1.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                tuple(),
                {"x": (1.0, 2.0), "y": 1.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                tuple(),
                {"y": (1.0, 2.0)},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (1.0, 2.0),
                tuple(),
                {"y": [1.0, 2.0]},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            ((0.0, 1.0), tuple(), {"r": 1, "phi": np.pi / 2}),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"r": 1},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"phi": np.pi / 2},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"r": 1, "x": 0.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"phi": np.pi / 2, "x": 0.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"r": 1, "y": 0.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"r": 1, "phi": np.pi / 2, "y": 0.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"phi": np.pi / 2, "y": 0.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                (1.0,),
                {"r": 1, "phi": np.pi / 2},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"r": "foo,", "phi": 0.0},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
            pytest.param(
                (0.0, 1.0),
                tuple(),
                {"r": 1, "phi": "foo"},
                marks=pytest.mark.xfail(raises=VectorValueError, strict=True),
            ),
        ],
    )
    def test_vector_init(self, expected, args, kwargs):
        v = Vector(*args, **kwargs)
        assert np.allclose(v._vector, expected)

        v_dim = DimVector(
            *tuple(val * U_("µm") for val in args),
            **{
                key: (val * U_("µm") if key != "phi" else val)
                for key, val in kwargs.items()
            }
        )
        assert np.allclose(v_dim._vector, expected * U_("µm"))

    @pytest.mark.parametrize(
        "expected, args, kwargs",
        [
            ((1.0, 0.001) * U_("µm"), (1 * U_("µm"), 1 * U_("nm")), {}),
            (
                (1.0, 2.0) * U_("µm"),
                (
                    Vector(
                        1.0,
                        2,
                    )
                    * U_("µm"),
                ),
                {},
            ),
            (
                (1.0, 2.0) * U_("µm"),
                (
                    U_("µm")
                    * Vector(
                        1.0,
                        2,
                    ),
                ),
                {},
            ),
        ],
    )
    def test_special_dim_vector_init(self, expected, args, kwargs):
        v_dim = DimVector(*args, **kwargs)
        assert np.allclose(v_dim._vector, expected)

    def test_vector_to_dim_vector(self):
        v = Vector(1, 2)

        dim_vec = v * U_("µm")
        assert isinstance(dim_vec, DimVector)

        dim_vec2 = DimVector(U_("µm") * v)
        assert isinstance(dim_vec2, DimVector)

    def test_dim_vector_special_methods(self):
        vec = Vector(2, 2)
        dim_vector: DimVector = vec * U_("µm")

        assert np.allclose(dim_vector.vector._vector, vec._vector)
        assert dim_vector.unit == U_("µm")

        assert np.allclose(
            dim_vector.normalized()._vector,
            (Vector(np.sqrt(2) / 2, np.sqrt(2) / 2) * U_("µm"))._vector,
        )

    def test__getitem__(self):
        for vec, mult in [
            (Vector(1.0, 2.0), 1.0),
            (Vector(1.0, 2.0) * U_("µm"), U_("µm")),
        ]:
            assert vec[0] == 1.0 * mult
            assert vec[1] == 2.0 * mult
            assert vec[:1] == [1.0 * mult]
            assert vec[1:] == [2.0 * mult]
            assert vec[:] == [1.0 * mult, 2.0 * mult]

        vec = Vector(1.0, 2.0)
        with pytest.raises(TypeError):
            vec["x"]

    def test__len__(self):
        assert len(Vector()) == 2
        assert len(DimVector()) == 2

    def test__array__(self):
        for v in [Vector(1.0, 2.0), DimVector([1.0, 2.0] * U_("µm"))]:
            v_array = np.asarray(v)
            assert isinstance(v_array, np.ndarray)
            assert np.allclose(v_array, (1.0, 2.0))

    def test_x_y(self):
        for vec, mult in [
            (Vector(1.0, 2.0), 1.0),
            (Vector(1.0, 2.0) * U_("µm"), U_("µm")),
        ]:
            assert vec.x == 1.0 * mult
            assert vec.y == 2.0 * mult

    def test_r_phi(self):
        for vec, r, phi in [
            (Vector(0.0, -2.0), 2.0, -np.pi / 2),
            (Vector(0.0, -2.0) * U_("µm"), 2 * U_("µm"), -np.pi / 2),
            (Vector(0.0, 2.0), 2.0, np.pi / 2),
            (Vector(0.0, 2.0) * U_("µm"), 2 * U_("µm"), np.pi / 2),
            (
                Vector(
                    0.0,
                    0,
                ),
                0.0,
                0.0,
            ),
            (Vector(0.0, 0.0) * U_("µm"), 0.0 * U_("µm"), 0.0),
        ]:
            assert np.isclose(vec.r, r)
            assert np.isclose(vec.phi, phi)

    def test_length_mag_magnitude(self):
        for v in [(0, 0), (1, 0), (3.76, 4.8), (-1.2, 2.4)]:
            length = np.linalg.norm(v)
            vec = Vector(v)
            assert np.isclose(vec.length, length)
            assert np.isclose(vec.mag, length)
            assert np.isclose(vec.magnitude, length)

            dim_vec = vec * U_("µm")
            assert np.isclose(dim_vec.length, length * U_("µm"))
            assert np.isclose(dim_vec.mag, length * U_("µm"))
            assert np.isclose(dim_vec.magnitude, length * U_("µm"))

    def test_angle_about_x_axis(self):
        for v, angle in [
            ((1, 0), 0.0),
            ((-1, 0), np.pi),
            ((1, 1), 1 / 4 * np.pi),
            ((-1, 1), 3 / 4 * np.pi),
            ((-1, -1), 5 / 4 * np.pi),
            ((1, -1), 7 / 4 * np.pi),
        ]:
            assert np.isclose(Vector(v).angle_about_x_axis, angle)
            assert np.isclose(DimVector(v * U_("µm")).angle_about_x_axis, angle)

        with pytest.raises(ValueError):
            Vector((0, 0)).angle_about_x_axis

        with pytest.raises(ValueError):
            DimVector((0, 0) * U_("µm")).angle_about_x_axis

    def test_close_to(self):
        vec = Vector(1, 2)
        dim_vec = vec * U_("µm")
        for other in [(1, 2), [1, 2], np.array([1, 2]), vec]:
            assert vec.close_to(other)
            assert dim_vec.close_to(other * U_("µm"))

            assert not (-vec).close_to(other)
            assert not (-dim_vec).close_to(other * U_("µm"))

    def test_normalized(self):
        a = -10
        b = 10
        N = 1000

        for v in (b - a) * np.random.random(size=(N, 2)) + a:
            assert np.allclose(Vector(v).normalized()._vector, v / np.linalg.norm(v))
            assert np.allclose(
                (Vector(v) * U_("µm")).normalized()._vector,
                v / np.linalg.norm(v) * U_("µm"),
            )

    def test_normalized_to(self):
        a = -10
        b = 10
        N = 1000

        for v in (b - a) * np.random.random(size=(N, 3)) + a:
            assert np.allclose(
                Vector(v[:2]).normalized_to(v[2])._vector,
                v[2] * v[:2] / np.linalg.norm(v[:2]),
            )
            assert np.allclose(
                (Vector(v[:2]) * U_("µm")).normalized_to(v[2])._vector,
                v[2] * v[:2] / np.linalg.norm(v[:2]) * U_("µm"),
            )

    def test_rotated(self):
        vec = Vector(2, 0)
        dim_vec = Vector(2, 0) * U_("µm")
        for angle in np.linspace(-np.pi, np.pi, 9):
            vec_rot = vec.rotated(angle)
            assert np.isclose(vec_rot.phi, angle)
            assert np.isclose(vec_rot.r, 2)

            dim_vec_rot = dim_vec.rotated(angle)
            assert np.isclose(dim_vec_rot.phi, angle)
            assert np.isclose(dim_vec_rot.r, 2 * U_("µm"))

        for angle in np.linspace(-np.pi, np.pi, 9):
            vec_rot = vec.rotated(angle, origin=(1, 0))

            # check if rotated point lies on the circle with radius 1 around (1, 0)
            assert np.isclose((vec_rot.x - 1) ** 2 + vec_rot.y**2, 1, atol=1e-9)

            dim_vec_rot = dim_vec.rotated(angle, origin=(1000, 0) * U_("nm"))
            assert np.isclose(
                (dim_vec_rot.x - 1 * U_("µm")) ** 2 + dim_vec_rot.y**2,
                1 * U_("µm**2"),
            )

    def test_mirrored(self):
        vec = Vector(2, 0)
        dim_vec = vec * U_("µm")

        for mirror_axis, rot_angle in [
            ((3, 3), np.pi / 2),
            ((-4, 4), 3 / 2 * np.pi),
            ((-5, -5), np.pi / 2),
            ((6, -6), 3 / 2 * np.pi),
        ]:
            mirrored = vec.mirrored(mirror_axis)
            assert np.isclose(mirrored.angle_about_x_axis, rot_angle)

            dim_mirrored = dim_vec.mirrored(mirror_axis * U_("µm"))
            assert np.isclose(dim_mirrored.angle_about_x_axis, rot_angle)

    def test_projected(self):
        vec = Vector(2, 2)
        assert np.isclose(vec.projected((1, 1)).r, np.sqrt(2))

        dim_vec = Vector(2, 2) * U_("µm")
        assert np.isclose(
            dim_vec.projected((1000, 1000) * U_("nm")).r, np.sqrt(2) * U_("µm")
        )

    def test_dot(self):
        assert np.isclose(Vector(1, 2).dot((3, 4)), 11)
        assert np.isclose(
            DimVector((1, 2) * U_("µm")).dot((3000, 4000) * U_("nm")), 11 * U_("µm**2")
        )

    def test__add____radd__(self):
        assert (Vector(1, 1) + (1, 1)).close_to((2, 2))
        assert ((1, 1) + Vector(1, 1)).close_to((2, 2))

        assert (DimVector((1, 1) * U_("µm")) + (1, 1) * U_("µm")).close_to(
            (2, 2) * U_("µm")
        )
        # TODO: this is not working. ask the pint people about this!
        # assert ((1, 1) * U_('µm') + DimVector((1, 1) * U_('µm'))).close_to((2, 2) * U_('µm'))

    def test__sub____rsub__(self):
        assert (Vector(1, 1) - (1, 1)).close_to((0, 0))
        assert ((1, 1) - Vector(1, 1)).close_to((0, 0))

        assert (DimVector((1, 1) * U_("µm")) - (1, 1) * U_("µm")).close_to(
            (0, 0) * U_("µm")
        )
        # TODO: this is not working. ask the pint people about this!
        # assert ((1, 1) * U_('µm') - DimVector((1, 1) * U_('µm'))).close_to((0, 0) * U_('µm'))

    def test__mul____rmul__(self):
        assert (Vector(1, -1) * 2).close_to((2, -2))
        assert (2 * Vector(1, -1)).close_to((2, -2))

        assert (DimVector((1, -1) * U_("µm") * 2)).close_to((2, -2) * U_("µm"))
        assert (2 * DimVector((1, -1) * U_("µm"))).close_to((2, -2) * U_("µm"))

    def test__truediv__(self):
        assert (Vector(1, -1) / 2).close_to((0.5, -0.5))
        assert (DimVector((1, -1) * U_("µm")) / 2).close_to((0.5, -0.5) * U_("µm"))

    def test__neg__(self):
        assert (-Vector(1, -1)).close_to((-1, 1))
        assert (-DimVector((1, -1) * U_("µm"))).close_to((-1, 1) * U_("µm"))

    @pytest.mark.parametrize(
        "expected, vec_1, vec_2",
        [
            (0, Vector(1, 0), Vector(2, 0)),
            (np.pi, Vector(1, 0), Vector(-1, 0)),
            (np.pi / 4, Vector(1, 0), Vector(1, 1)),
            (0, Vector(1, 0) * U_("µm"), Vector(2, 0) * U_("µm")),
            (np.pi, Vector(1, 0) * U_("µm"), Vector(-1, 0) * U_("µm")),
            (np.pi / 4, Vector(1, 0) * U_("µm"), Vector(1, 1) * U_("µm")),
            (0, Vector(1, 0) * U_("µm"), Vector(2, 0)),
            (np.pi, Vector(1, 0), Vector(-1, 0) * U_("µm")),
            (np.pi / 4, Vector(1, 0) * U_("µm"), Vector(1, 1)),
            (np.pi / 4, (1, 0), (1, 1)),
            (np.pi / 4, (1, 0), Vector(1, 1)),
            (np.pi / 4, (1, 0), Vector(1, 1) * U_("µm")),
            (np.pi / 4, Vector(1, 1), (1, 0)),
            (np.pi / 4, Vector(1, 1) * U_("µm"), (1, 0)),
            pytest.param(
                np.pi / 4,
                (1,),
                (1, 1),
                marks=pytest.mark.xfail(raises=ValueError, strict=True),
            ),
            pytest.param(
                np.pi / 4,
                (1, 0),
                (1,),
                marks=pytest.mark.xfail(raises=ValueError, strict=True),
            ),
        ],
    )
    def test_angle_between(self, expected, vec_1, vec_2):
        assert np.isclose(angle_between(vec_1, vec_2), expected)
