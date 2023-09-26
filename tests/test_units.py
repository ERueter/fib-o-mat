import itertools


import numpy as np

from fibomat.units import ureg, Q_, U_, UnitType, QuantityType, scale_factor, scale_to, has_length_dim, has_time_dim

class TestUnits:
    def test_ureg(self):
        # test only if these things are not None
        assert ureg
        assert Q_
        assert U_
        assert UnitType
        assert QuantityType

    def test_ions(self):
        q = Q_('1 ions')
        assert q == Q_('1 elementary_charge')

        # TODO: make this work
        # assert Q_('1 ions * s').check('[spotdose]')
        # assert Q_('1 ions * s / m').check('[linedose]')
        # assert Q_('1 ions * s / m**2').check('[areadose]')

    def test_scaling(self):
        for s in itertools.combinations_with_replacement([1., 2., 3.], 2):
            assert np.isclose(scale_factor(s[0] * Q_('1 cm'), s[1] * Q_('1 m')), 100)
            assert np.isclose(scale_factor(s[0] * Q_('1 cm'), s[1] * U_('1 m')), 100)
            assert np.isclose(scale_factor(s[0] * U_('1 cm'), s[1] * Q_('1 m')), 100)
            assert np.isclose(scale_factor(s[0] * U_('1 cm'), s[1] * U_('1 m')), 100)

            assert np.isclose(scale_to(U_('m'), s[0] * Q_('1 cm')), s[0] / 100)
            assert np.isclose(scale_to(s[1] * Q_('1 m'), s[0] * Q_('1 cm')), s[0] / 100)

    def test_has_dim(self):
        assert has_length_dim(U_('µm'))
        assert has_time_dim(U_('µs'))
