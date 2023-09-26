from typing import List

import pytest

from fibomat.sample import Sample
from fibomat.site import Site
from fibomat.linalg import DimVector, Vector
from fibomat.units import U_
from fibomat.shapes import Spot, DimShape
from fibomat.backend import BackendBase, registry
from fibomat.default_backends import BokehBackend


class FakeBackend(BackendBase):

    name = 'FakeBackend'

    def __init__(self, **kwargs):
        super().__init__()

        self.exported_sites = []
        self.kwargs = kwargs

    def process_site(self, new_site: Site) -> None:
        self.exported_sites.append(new_site)


class TestSample:
    def test_init(self):
        sample = Sample(description='foo')

        assert sample.description == 'foo'

    def test_add_site(self):
        sample = Sample()

        site_1 = Site(DimVector())
        sample.add_site(site_1)

        site_2 = Site((1, 1) * U_('µm'))
        sample += site_2

        site_3 = sample.create_site((2, 2) * U_('µm'))
        assert site_3
        site_4 = sample.create_site((3, 3) * U_('µm'), DimVector(), description='bar')
        assert site_4

        assert len(sample._sites) == 4
        assert sample._sites[0].center == (0, 0) * U_('µm')
        assert sample._sites[1].center == (1, 1) * U_('µm')
        assert sample._sites[2].center == (2, 2) * U_('µm')
        assert sample._sites[3].center == (3, 3) * U_('µm')

    @pytest.mark.parametrize("exporter_class", [FakeBackend, FakeBackend.name])
    def test_export(self, exporter_class):
        registry.register(FakeBackend, FakeBackend.name)

        sample = Sample()
        site_1 = sample.create_site(Vector(1, 1) * U_('µm'))
        site_2 = sample.create_site(Vector(2, 2) * U_('µm'))

        exported: FakeBackend = sample.export(exporter_class, a=1, b='foo')

        assert len(exported.exported_sites) == 2
        assert exported.exported_sites[0] == site_1
        assert exported.exported_sites[1] == site_2
        assert exported.kwargs['a'] == 1
        assert exported.kwargs['b'] == 'foo'

    @pytest.mark.parametrize("exporter_class", [FakeBackend, FakeBackend.name])
    def test_multi_export(self, exporter_class):
        registry.register(FakeBackend, FakeBackend.name)

        sample = Sample()
        site_1 = sample.create_site(Vector(1, 1) * U_('µm'))
        site_2 = sample.create_site(Vector(2, 2) * U_('µm'))
        sites = [site_1, site_2]

        exported: List[FakeBackend] = sample.export_multi(exporter_class, a=1, b='foo')

        assert len(exported) == 2

        for i, part_export in enumerate(exported):
            assert len(part_export.exported_sites) == 1
            assert part_export.exported_sites[0] == sites[i]
            assert part_export.kwargs['a'] == 1
            assert part_export.kwargs['b'] == 'foo'

    @pytest.mark.parametrize(
        "show_plot, filepath",
        [(False, None), (True, None), (False, 'foo'), (True, 'bar')])
    def test_plot_export(self, monkeypatch, show_plot, filepath):
        def mk_init(self, **kwargs):
            self.kwargs = kwargs
            self.sites = []
            self.annotations = []

            self.plotted = False
            self.shown = False
            self.filepath = None

        def mk_process_site(self, site):
            self.sites.append(site)

        def mk_process_pattern(self, pattern):
            self.annotations.append(pattern)

        def mk_plot(self):
            self.plotted = True

        def mk_show(self):
            self.shown = True

        def mk_save(self, fp):
            self.filepath = fp

        monkeypatch.setattr(BokehBackend, '__init__', mk_init)
        monkeypatch.setattr(BokehBackend, 'process_site', mk_process_site)
        monkeypatch.setattr(BokehBackend, 'process_pattern', mk_process_pattern)
        monkeypatch.setattr(BokehBackend, 'plot', mk_plot)
        monkeypatch.setattr(BokehBackend, 'show', mk_show)
        monkeypatch.setattr(BokehBackend, 'save', mk_save)

        sample = Sample()
        site_1 = sample.create_site(Vector(1, 1) * U_('µm'))
        site_2 = sample.create_site(Vector(2, 2) * U_('µm'))
        sites = [site_1, site_2]

        sample.add_annotation(Spot((0, 0)) * U_('µm'))
        sample.add_annotation(Spot((1, 1)) * U_('µm'), filled=True, color='red', description='baz')

        plot = sample.plot(show=show_plot, filename=filepath, a=1, b='baz')

        assert plot.plotted

        if show_plot:
            assert plot.shown

        if filepath:
            assert plot.filepath == filepath

        for i, site in enumerate(plot.sites):
            assert site == sites[i]

        assert len(plot.annotations) == 2
        assert plot.annotations[0].dim_shape.shape.center == (0, 0)
        assert plot.annotations[0].dim_shape.unit == U_('µm')
        assert plot.annotations[0].raster_style.dimension == 1
        assert plot.annotations[0].kwargs['_annotation'] is True
        assert plot.annotations[0].kwargs['_color'] is None
        assert plot.annotations[0].description is None

        assert plot.annotations[1].dim_shape.shape.center == (1, 1)
        assert plot.annotations[1].dim_shape.unit == U_('µm')
        assert plot.annotations[1].raster_style.dimension == 2
        assert plot.annotations[1].kwargs['_annotation'] is True
        assert plot.annotations[1].kwargs['_color'] == 'red'
        assert plot.annotations[1].description == 'baz'

        assert plot.kwargs['a'] == 1
        assert plot.kwargs['b'] == 'baz'

    def test_annotations(self):
        sample = Sample()

        sample.add_annotation(Spot((0, 0)) * U_('µm'))
        sample.add_annotation(Spot((1, 1)) * U_('µm'), filled=True, color='red', description='baz')

        assert len(sample._annotations) == 2
        assert sample._annotations[0].dim_shape.shape.center == (0, 0)
        assert sample._annotations[0].dim_shape.unit == U_('µm')
        assert sample._annotations[0].filled is False
        assert sample._annotations[0].color is None
        assert sample._annotations[0].description is None

        assert sample._annotations[1].dim_shape.shape.center == (1, 1)
        assert sample._annotations[1].dim_shape.unit == U_('µm')
        assert sample._annotations[1].filled is True
        assert sample._annotations[1].color == 'red'
        assert sample._annotations[1].description == 'baz'

