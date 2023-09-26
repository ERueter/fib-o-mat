import pytest

from fibomat.mill import Mill, MillBase, SpecialMill
from fibomat.units import Q_


class TestMill:
    def test_millbase(self):
        mill = MillBase(a=1, b=2)

        # assert mill.a == 1
        # assert mill.b == 2

        assert mill['a'] == 1
        assert mill['b'] == 2

        with pytest.raises(KeyError):
            mill['c']

        # with pytest.raises(KeyError):
        #     mill.c
        #
        # with pytest.raises(TypeError):
        #     mill.a = 2

        with pytest.raises(TypeError):
            mill['a'] = 2

    def test_mill(self):
        mill = Mill(Q_('1 s'), 2)

        assert mill['dwell_time'] == Q_('1 s')
        assert mill['repeats'] == 2

        assert mill.dwell_time == Q_('1 s')
        assert mill.repeats == 2

        with pytest.raises(TypeError):
            Mill('foo', 2)

        with pytest.raises(ValueError):
            Mill(Q_('1 pA'), 2)

        with pytest.raises(TypeError):
            Mill(Q_('1 s'), 'foo')

        with pytest.raises(ValueError):
            Mill(Q_('1 s'), 0)

    def test_specialmill(self):
        sm = SpecialMill(a=1, b=2)

        assert sm['a'] == 1
        assert sm['b'] == 2
