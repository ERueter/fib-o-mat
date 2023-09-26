from fibomat.describable import Describable


class TestDescribable:
    def test_props(self):
        d_1 = Describable()
        assert not d_1.description

        d_2 = Describable(description='foo')
        assert d_2.description == 'foo'

        d_3 = Describable('bar')
        assert d_3.description == 'bar'

    def test_copy_with_new_descr(self):
        d = Describable(description='foo')
        d_copy = d.with_changed_description('bar')

        assert d.description == 'foo'
        assert d_copy.description == 'bar'

