"""Tests of the NPVE single site backend and of the deflection lists of both NPVE shape backends.

The reference files in ``data/npve`` were written by NPVE: ``testing.nps`` contains a shape, ``test.nps`` and
``my_file.dat`` a deflection list in a binary file and ``test_2.nps`` the same list embedded in the file.
"""
import pathlib
import re
import struct
import warnings

import numpy as np
import pytest
import xmltodict

from fibomat.default_backends.npve import (
    NPVEMill, NPVESingleSite, NPVEStepAndRepeat, StepAndRepeatBackend
)
from fibomat.default_backends.npve.step_and_repeat.deflection_list import DeflectionList
from fibomat.arrangements import Lattice
from fibomat.layout import Layout, Pattern
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d, zero_d
from fibomat.shapes import Circle, Line, RasterizedPoints, Rect, Spot
from fibomat.units import DimensionError, unit

from sar_corpus import lines, repeats_mill, um


DATA = pathlib.Path(__file__).parent / 'data' / 'npve'
BACKENDS = [NPVESingleSite, NPVEStepAndRepeat]


def read_reference(name):
    return xmltodict.parse((DATA / name).read_text(encoding='iso-8859-1'), force_list=('FIBShape', 'Pt'))


def reference_points():
    """Points of ``my_file.dat`` as (n, 3) array."""
    data = (DATA / 'my_file.dat').read_bytes()
    points = np.frombuffer(data, dtype=[('x', '<f4'), ('y', '<f4'), ('repeat', '<u4')], offset=8)
    return np.column_stack((points['x'], points['y'], points['repeat'])).astype(float)


def deflection_layout(points, mill=None, site_center=(0, 0)):
    layout = Layout()
    layout.create_site(um(*site_center), um(10, 10)).create_pattern(
        RasterizedPoints(points, False) * unit('µm'), mill or repeats_mill(1), zero_d.PreRasterized()
    )
    return layout


def parse(backend):
    return xmltodict.parse(backend._to_str(), force_list=('FIBShape', 'Pt'))  # pylint: disable=protected-access


def shapes_of(document):
    if 'nps' in document:
        return document['nps']['FIBShapes']['FIBShape']
    return document['StepAndRepeatPatterning']['Sites']['Site']['Shapes']['FIBShape']


def structure(node):
    """Names of all elements (recursively, in the order of the file) of an XML document."""
    if isinstance(node, dict):
        return [(key, structure(value)) for key, value in node.items()]
    if isinstance(node, list):
        return structure(node[0])
    return None


SMALL_POINTS = np.array([[0., 1., 1.], [2., 3., 2.], [-1., 0.5, 4.]])


class TestFileFormat:
    def test_header_is_identical_to_reference(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Rect(5, 6) * unit('µm'), repeats_mill(), lines())

        generated = parse(layout.export(NPVESingleSite))['nps']
        reference = read_reference('testing.nps')['nps']

        for key in ('Version', 'Publisher', 'Type', 'Extension', 'MicroscopePersonality', 'ActiveAperture'):
            assert generated[key] == reference[key]
        assert list(generated) == list(reference)

    def test_shape_elements_are_the_same_as_in_the_reference(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Rect(5, 6) * unit('µm'), repeats_mill(), lines())

        generated = shapes_of(parse(layout.export(NPVESingleSite)))[0]
        reference = shapes_of(read_reference('testing.nps'))[0]

        # the nodes of a polygon have the same form, their number is not checked by `structure`
        assert structure(generated) == structure(reference)
        assert generated['Mill']['OperationID'] == reference['Mill']['OperationID']
        assert generated['Angle'] == '0.0'

    @pytest.mark.parametrize('reference_name, max_embedded', [('test.nps', 0), ('test_2.nps', 10000)])
    def test_deflection_list_elements_are_the_same_as_in_the_reference(self, reference_name, max_embedded):
        layout = deflection_layout(SMALL_POINTS)
        generated = shapes_of(parse(layout.export(NPVESingleSite, max_embedded_points=max_embedded)))[0]
        reference = shapes_of(read_reference(reference_name))[0]

        assert structure(generated) == structure(reference)
        assert generated['Class'] == 'TDeflectionListShape'
        assert generated['ShapeName'] == 'Deflection List 1'
        for key in ('Outlined', 'NodeStyle', 'StrokeStyle', 'PenAlignment', 'Direction'):
            assert generated['Outline'][key] == reference['Outline'][key]
        assert generated['Nodes'] is None

    def test_iso_8859_1(self, tmp_path):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Rect(5, 6) * unit('µm'), repeats_mill(), lines())
        layout.export(NPVESingleSite, aperture='[2] He 10 µm').save(tmp_path / 'a.nps')

        data = (tmp_path / 'a.nps').read_bytes()
        assert data.startswith(b'<?xml version="1.0" encoding="iso-8859-1"?>')
        assert '[2] He 10 µm'.encode('iso-8859-1') in data


class TestSingleSite:
    def test_only_one_site(self):
        layout = Layout()
        for x in (0, 10):
            layout.create_site(um(x, 0)).create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
        with pytest.raises(ValueError, match='only one site'):
            layout.export(NPVESingleSite)

    def test_export_multi_writes_one_backend_per_site(self):
        layout = Layout()
        for x in (0, 10):
            layout.create_site(um(x, 0)).create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
        assert len(layout.export_multi(NPVESingleSite)) == 2

    def test_empty_sites_are_ignored(self):
        layout = Layout()
        layout.create_site(um(5, 0), um(1, 1))
        layout.create_site(um(0, 0)).create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
        layout.create_site(um(6, 0), um(1, 1))
        assert len(shapes_of(parse(layout.export(NPVESingleSite)))) == 1

    def test_no_shapes(self):
        with pytest.raises(RuntimeError, match='any shapes'):
            Layout().export(NPVESingleSite).save('never_written.nps')
        assert not pathlib.Path('never_written.nps').exists()

    def test_ids_start_at_one(self):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
        site.create_pattern(Circle(1) * unit('µm'), repeats_mill(), lines())
        assert [shape['DisplayID'] for shape in shapes_of(parse(layout.export(NPVESingleSite)))] == ['1', '2']

    def test_header_options(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
        document = parse(layout.export(NPVESingleSite, system_name='A', beam='B', aperture='C'))['nps']
        assert document['MicroscopePersonality']['BeamController']['Info'] == {'SystemName': 'A', 'Beam': 'B'}
        assert document['ActiveAperture']['Name'] == 'C'

    def test_shapes_are_relative_to_the_site(self):
        points = np.array([[1., 2., 1.], [3., 4., 1.]])
        shape = shapes_of(parse(deflection_layout(points, site_center=(100, 100)).export(NPVESingleSite)))[0]
        first = shape['DeflectionListData']['PointsList']['SubList']['Pt'][0]
        assert (float(first['X']), float(first['Y'])) == (1., 2.)

    @pytest.mark.parametrize('shape, style', [
        (Spot() , zero_d.SingleSpot()),
        (Line((0, 0), (1, 1)), one_d.Curve(0.1 * unit('µm'), ScanSequence.CONSECUTIVE)),
        (Circle(1), lines()),
    ])
    def test_same_shapes_as_step_and_repeat(self, shape, style):
        def export(backend):
            layout = Layout()
            layout.create_site(um(0, 0), um(4, 4)).create_pattern(shape * unit('µm'), repeats_mill(), style)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)  # spot dose
                return shapes_of(parse(layout.export(backend)))[0]

        single, step_and_repeat = export(NPVESingleSite), export(NPVEStepAndRepeat)
        for key in ('Class', 'Nodes', 'RotationCenter', 'Outline'):
            assert single[key] == step_and_repeat[key]


@pytest.mark.parametrize('backend', BACKENDS)
class TestDeflectionLists:
    def test_embedded(self, backend):
        shape = shapes_of(parse(deflection_layout(SMALL_POINTS).export(backend)))[0]
        points_list = shape['DeflectionListData']['PointsList']

        assert 'DataFilename' not in points_list
        assert (points_list['Count'], points_list['RepCount']) == ('3', '3')
        assert points_list['Units'] == '0'
        assert [(float(p['X']), float(p['Y']), int(p['Repeat'])) for p in points_list['SubList']['Pt']] == \
            [tuple(point) for point in SMALL_POINTS]
        box = points_list['RawBoundingBox']
        assert (float(box['XMin']), float(box['XMax']), float(box['YMin']), float(box['YMax'])) == (-1., 2., .5, 3.)

    def test_external(self, backend, tmp_path):
        exported = deflection_layout(SMALL_POINTS).export(backend, max_embedded_points=2)
        exported.save(tmp_path / 'file.xml')

        document = xmltodict.parse((tmp_path / 'file.xml').read_text(encoding='iso-8859-1'), force_list=('FIBShape',))
        points_list = shapes_of(document)[0]['DeflectionListData']['PointsList']
        assert 'SubList' not in points_list
        assert points_list['Count'] == '3'

        data = (tmp_path / points_list['DataFilename']).read_bytes()
        assert struct.unpack('<2I', data[:8]) == (1, 3)
        assert len(data) == 8 + 3 * 12
        assert struct.unpack('<ffI', data[8:20]) == (0., 1., 1)

    def test_cutoff(self, backend):
        for n_points, embedded in ((5, True), (6, False)):
            points = np.column_stack((np.arange(n_points, dtype=float), np.zeros(n_points), np.ones(n_points)))
            exported = deflection_layout(points).export(backend, max_embedded_points=5)
            assert ('SubList' in shapes_of(parse(exported))[0]['DeflectionListData']['PointsList']) == embedded

    def test_default_cutoff(self, backend):
        n_points = 5001
        points = np.column_stack((np.arange(n_points, dtype=float), np.zeros(n_points), np.ones(n_points)))
        shape = shapes_of(parse(deflection_layout(points).export(backend)))[0]
        assert 'DataFilename' in shape['DeflectionListData']['PointsList']

    def test_embedded_and_external_contain_the_same_points(self, backend, tmp_path):
        points = reference_points()
        embedded = deflection_layout(points).export(backend, max_embedded_points=10000)
        external = deflection_layout(points).export(backend, max_embedded_points=0)
        external.save(tmp_path / 'file.xml')

        points_list = shapes_of(parse(embedded))[0]['DeflectionListData']['PointsList']
        from_xml = np.array([[float(p['X']), float(p['Y']), int(p['Repeat'])] for p in points_list['SubList']['Pt']])
        assert np.array_equal(from_xml, points)

        name = shapes_of(parse(external))[0]['DeflectionListData']['PointsList']['DataFilename']
        assert (tmp_path / name).read_bytes() == (DATA / 'my_file.dat').read_bytes()

    def test_units_are_converted(self, backend):
        layout = Layout()
        layout.create_site(um(0, 0), um(10, 10)).create_pattern(
            RasterizedPoints(np.array([[1000., 0., 1.], [0., 500., 1.]]), False) * unit('nm'), repeats_mill(),
            zero_d.PreRasterized()
        )
        shape = shapes_of(parse(layout.export(backend)))[0]
        first = shape['DeflectionListData']['PointsList']['SubList']['Pt'][0]
        assert float(first['X']) == pytest.approx(1.)

    def test_mill(self, backend):
        mill = NPVEMill(dwell_time=1.5 * unit('µs'), repeats=3)
        shape = shapes_of(parse(deflection_layout(SMALL_POINTS, mill).export(backend)))[0]
        assert float(shape['Mill']['DwellTime']) == 1.5
        assert shape['Mill']['NumFrames'] == '3'
        assert shape['Mill']['TargetMode'] == '3'

    @pytest.mark.parametrize('weight', [0.5, 0., -1.])
    def test_weights_must_be_positive_integers(self, backend, weight):
        points = np.array([[0., 0., 1.], [1., 1., weight]])
        with pytest.raises(ValueError, match='positive integers'):
            deflection_layout(points).export(backend)

    def test_dose_is_not_supported(self, backend):
        mill = NPVEMill(dwell_time=1. * unit('µs'), dose=1. * unit('nC / µm**2'))
        with pytest.raises(ValueError, match='repeats'):
            deflection_layout(SMALL_POINTS, mill).export(backend)

    def test_other_mills_are_rejected(self, backend):
        with pytest.raises(TypeError, match='NPVEMill'):
            deflection_layout(SMALL_POINTS, Mill(1. * unit('µs'), 1)).export(backend)

    def test_needs_prerasterized(self, backend):
        layout = Layout()
        layout.create_site(um(0, 0), um(4, 4)).create_pattern(
            RasterizedPoints(SMALL_POINTS, False) * unit('µm'), repeats_mill(),
            one_d.Curve(0.1 * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        with pytest.raises(TypeError, match='PreRasterized'):
            layout.export(backend)

    def test_prerasterized_is_only_for_deflection_lists(self, backend):
        layout = Layout()
        layout.create_site(um(0, 0), um(4, 4)).create_pattern(
            Rect(1, 1) * unit('µm'), repeats_mill(), zero_d.PreRasterized()
        )
        with pytest.raises(TypeError, match='Unsupported raster style'):
            layout.export(backend)

    def test_several_external_lists_get_their_own_files(self, backend, tmp_path):
        layout = Layout()
        site = layout.create_site(um(0, 0), um(10, 10))
        for i in range(2):
            site.create_pattern(
                RasterizedPoints(SMALL_POINTS + [i, 0, 0], False) * unit('µm'), repeats_mill(), zero_d.PreRasterized()
            )
        exported = layout.export(backend, max_embedded_points=0)
        exported.save(tmp_path / 'file.xml')
        names = [shape['DeflectionListData']['PointsList']['DataFilename'] for shape in shapes_of(parse(exported))]
        assert len(set(names)) == 2
        assert all(re.fullmatch(r'file_\d+\.dat', name) for name in names)
        assert all((tmp_path / name).exists() for name in names)

    def test_invalid_max_embedded_points(self, backend):
        with pytest.raises(ValueError, match='max_embedded_points'):
            backend(max_embedded_points=-1)


CURVE = one_d.Curve(0.1 * unit('µm'), ScanSequence.CONSECUTIVE)


def dose_layout(shape, dose, style, description=None):
    layout = Layout()
    layout.create_site(um(0, 0), um(4, 4)).create_pattern(
        shape * unit('µm'), NPVEMill(dwell_time=1. * unit('µs'), dose=dose), style, description=description
    )
    return layout


@pytest.mark.parametrize('backend', BACKENDS)
class TestValidation:
    @pytest.mark.parametrize('shape, style, dose, expected', [
        (Line((0, 0), (1, 1)), CURVE, 1. * unit('ions / nm**2'), 'line dose'),
        (Line((0, 0), (1, 1)), CURVE, 1. * unit('ions'), 'line dose'),
        (Rect(1, 1), lines(), 1. * unit('nC / µm'), 'area dose'),
        (Rect(1, 1), lines(), 1. * unit('ions'), 'area dose'),
        (Spot(), zero_d.SingleSpot(), 1. * unit('ions / nm**2'), 'spot dose'),
        (Spot(), zero_d.SingleSpot(), 1. * unit('nC / µm'), 'spot dose'),
        (Rect(1, 1), CURVE, 1. * unit('ions / nm**2'), 'line dose'),
    ])
    def test_dose_must_fit_to_the_raster_style(self, backend, shape, style, dose, expected):
        with pytest.raises(DimensionError, match=expected):
            dose_layout(shape, dose, style).export(backend)

    @pytest.mark.parametrize('shape, style, dose', [
        (Line((0, 0), (1, 1)), CURVE, 1. * unit('nC / µm')),
        (Rect(1, 1), lines(), 1. * unit('ions / nm**2')),
        (Rect(1, 1), lines(), 1. * unit('nC / µm**2')),
        (Rect(1, 1), CURVE, 1. * unit('nC / µm')),
        (Spot(), zero_d.SingleSpot(), 1. * unit('ions')),
    ])
    def test_fitting_doses_are_accepted(self, backend, shape, style, dose):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # spot dose
            dose_layout(shape, dose, style).export(backend)

    def test_error_names_the_pattern(self, backend):
        with pytest.raises(DimensionError, match="'my line'.*Curve.*dimension 1.*line dose.*ions/nm"):
            dose_layout(Line((0, 0), (1, 1)), 1. * unit('ions / nm**2'), CURVE, 'my line').export(backend)

    def test_mills_with_repeats_are_not_checked(self, backend):
        layout = Layout()
        layout.create_site(um(0, 0), um(4, 4)).create_pattern(Line((0, 0), (1, 1)) * unit('µm'), repeats_mill(), CURVE)
        layout.export(backend)

    def test_arrangements_are_validated_shape_by_shape(self, backend):
        layout = Layout()
        layout.create_site(um(0, 0), um(9, 9)).create_pattern(
            Lattice.from_counts(2, 1, 2, 2, Line((0, 0), (1, 1))) * unit('µm'),
            NPVEMill(dwell_time=1. * unit('µs'), dose=1. * unit('ions')), CURVE
        )
        with pytest.raises(DimensionError, match='line dose'):
            layout.export(backend)

    def test_direct_call(self, backend):
        pattern = Pattern(
            Line((0, 0), (1, 1)) * unit('µm'), NPVEMill(dwell_time=1. * unit('µs'), dose=1. * unit('ions')), CURVE
        )
        with pytest.raises(DimensionError):
            backend().validate_pattern(pattern)


class TestStepAndRepeatDeflectionList:
    def test_several_sites_and_shared_patterns(self):
        layout = Layout()
        for x in (0, 10):
            layout.create_site(um(x, 0), um(10, 10)).create_pattern(
                RasterizedPoints(SMALL_POINTS, False) * unit('µm'), repeats_mill(), zero_d.PreRasterized()
            )
        document = xmltodict.parse(layout.export(NPVEStepAndRepeat)._to_str())  # pylint: disable=protected-access
        sites = document['StepAndRepeatPatterning']['Sites']['Site']
        assert len(sites) == 2
        assert all(site['Shapes']['FIBShape']['Class'] == 'TDeflectionListShape' for site in sites)

        shared = Layout()
        shared.create_site(um(0, 0), um(10, 10)).create_pattern(
            RasterizedPoints(SMALL_POINTS, False) * unit('µm'), repeats_mill(), zero_d.PreRasterized()
        )
        shared.create_site(um(10, 0), um(10, 10))
        document = xmltodict.parse(
            shared.export(NPVEStepAndRepeat, share_patterns=True)._to_str()  # pylint: disable=protected-access
        )
        assert document['StepAndRepeatPatterning']['SharedShapes']['FIBShape']['Class'] == 'TDeflectionListShape'


class TestDeprecatedName:
    def test_warns_and_behaves_like_the_new_class(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())

        with pytest.warns(DeprecationWarning, match='NPVEStepAndRepeat'):
            old = layout.export(StepAndRepeatBackend)
        assert isinstance(old, NPVEStepAndRepeat)
        assert old._to_str() == layout.export(NPVEStepAndRepeat)._to_str()  # pylint: disable=protected-access

    def test_new_name_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            NPVEStepAndRepeat()
            NPVESingleSite()


class TestDeflectionListClass:
    def test_empty(self):
        with pytest.raises(ValueError, match='at least one'):
            DeflectionList(np.empty((0, 2)), np.empty(0), 1, True)

    def test_not_finite(self):
        with pytest.raises(ValueError, match='finite'):
            DeflectionList(np.array([[np.nan, 0.]]), np.array([1.]), 1, True)

    def test_reference_file_is_reproduced(self):
        points = reference_points()
        deflection_list = DeflectionList(points[:, :2], points[:, 2], 1, False)
        assert deflection_list.to_bytes() == (DATA / 'my_file.dat').read_bytes()
