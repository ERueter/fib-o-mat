import argparse

import numpy as np

from fibomat import sample, linalg, units
from fibomat.shapes import Arc, ArcSpline, Polygon, RasterizedPoints
from fibomat.units import U_, Q_
from fibomat.pattern import Pattern
from fibomat.mill import Mill, GaussBeam
import fibomat.raster_styles as raster
from fibomat.default_backends import SpotListBackend
from fibomat.curve_tools import rasterize
from fibomat.optimize import optimze_rasterized_hybrid

from fibomat import utils
from typing import Dict, Any


class NPVETxt(SpotListBackend):
    name = 'NPVEtxt'

    def __init__(self, description=None):

        def _save_impl(filename: utils.PathLike, dwell_points: np.ndarray, parameters: Dict[str, Any]):
            if 'base_dwell_time' not in parameters or not parameters['base_dwell_time']:
                raise RuntimeError

            dwell_points[:, 2] /= parameters['base_dwell_time']

            fov = max(parameters["fov"].width, parameters["fov"].height)

            with open(filename, 'w') as fp:
                # header.write(fp)
                # fp.write('[Points]\n')
                fp.write('NPVE DEFLECTION LIST\n')
                fp.write('UNITS=MICRONS\n')
                fp.write('DWELL=0.1\n')
                fp.write(f'FOV={fov}\n')
                fp.write('START\n')
                np.savetxt(fp, dwell_points, "%.5f %.5f %d")
                fp.write('END\n')

        super().__init__(save_impl=_save_impl, base_dwell_time=Q_('0.1 µs'), length_unit=U_('µm'), time_unit=U_('µs'))


def make_oligomere_outline(where: str, corner_polygon: Polygon, mono_radius: float):
    """
    Creates a path connecting all monomers on the inside or outside (given by `where`).

    Args:
        where (string): `inside` or `outside`
        corner_polygon (shapes.Polygon): polygon defining the centers of the monomers
        mono_radius (float): radius of a single monomer (with gap)

    Returns:

    """

    if where == 'outer':
        fac = 1.
    elif where == 'inner':
        fac = -1.
    else:
        raise ValueError('unknown `where`.')

    arcs = []

    n_monomers = corner_polygon.n_points

    corner_points = corner_polygon.points

    for i in range(n_monomers):
        arc_start = (corner_points[i % n_monomers] + corner_points[(i + 1) % n_monomers]) / 2
        arc_end = (corner_points[(i + 1) % n_monomers] + corner_points[(i + 2) % n_monomers]) / 2
        arc_middle = corner_polygon.points[(i + 1) % n_monomers] + fac * linalg.Vector(
            corner_polygon.points[(i + 1) % n_monomers]
        ).normalized() * mono_radius

        arcs.append(Arc.from_points(arc_start, arc_middle, arc_end))

    return arcs


def make_oligomere_base_curves(corner_polygon: Polygon, mono_radius_with_gap: float, has_gap: bool):
    """
    Creates the base curves of the oligomere

    Args:
        corner_polygon:
        mono_radius_with_gap:
        has_gap:

    Returns:

    """
    assert corner_polygon.n_points % 2 == 0

    inner_arcs = make_oligomere_outline('inner', corner_polygon, mono_radius_with_gap)
    outer_arcs = make_oligomere_outline('outer', corner_polygon, mono_radius_with_gap)
    separators = []

    if not has_gap:
        curve = ArcSpline.from_segments([
            outer_arcs[1], inner_arcs[1].reversed(), outer_arcs[0].reversed(), inner_arcs[0]
        ])

        for rot_angle in np.linspace(0, 2*np.pi, int(corner_polygon.n_points / 2), endpoint=False):
            separators.append(curve.rotated(rot_angle))
    return {
        'inner': ArcSpline.from_segments(inner_arcs),
        'outer': ArcSpline.from_segments(outer_arcs),
        'separators': separators
    }


def main():
    # define command line arguments
    parser = argparse.ArgumentParser(description='Generate oligomere pattern')
    parser.add_argument('--monomers', dest='n_monomers', required=True, type=int, help='number of monomers')
    parser.add_argument('--radius', dest='mono_radius', required=True, type=float, help='radius of monomers')
    parser.add_argument('--gap', dest='mono_gap', default=0., type=float, help='gap between monomers')
    parser.add_argument('--inner-repeats', dest='inner_repeats', required=True, type=int, help='#repeats of inner part')
    parser.add_argument('--outer-repeats', dest='outer_repeats', required=True, type=int, help='#repeats of outer part')
    parser.add_argument('--mono-repeats', dest='mono_repeats', required=True, type=int, help='#repeats of monomer separating curves (only used if gap = 0.)')
    parser.add_argument('--pitch-inner', dest='pitch_inner', required=True, type=float, help='pitch in inner part')
    parser.add_argument('--pitch-outer', dest='pitch_outer', required=True, type=float, help='pitch in outer part and mono curves')
    parser.add_argument('--offset-delta', dest='offset_delta', required=True, type=float, help='global offset delta')
    parser.add_argument('--dwell-time', dest='dwell_time', required=True, type=float, help='global dwell_time')
    parser.add_argument('--outer-margin', dest='outer_margin', required=True, type=float, help='outer margin')

    args = parser.parse_args()

    # collect command line arguments
    n_monomers = args.n_monomers
    if n_monomers < 2:
        raise RuntimeError('n_monomers < 2')

    mono_radius = args.mono_radius
    if mono_radius <= 0.:
        raise RuntimeError('mono_radius <= 0.')

    mono_gap = args.mono_gap
    if mono_gap < 0.:
        raise RuntimeError('mono_gap < 0.')

    inner_repeats = args.inner_repeats
    if inner_repeats < 1:
        raise RuntimeError('inner_repeats < 1')

    outer_repeats = args.outer_repeats
    if outer_repeats < 1:
        raise RuntimeError('outer_repeats < 1')

    mono_repeats = args.mono_repeats
    if mono_repeats < 1:
        raise RuntimeError('mono_repeats < 1')

    pitch_inner = args.pitch_inner * U_('µm')
    if args.pitch_inner < 0.:
        raise RuntimeError('pitch_inner < 0.')

    pitch_outer = args.pitch_outer * U_('µm')
    if args.pitch_outer < 0.:
        raise RuntimeError('pitch_outer < 0.')

    offset_delta = args.offset_delta * U_('µm')
    if args.offset_delta < 0.:
        raise RuntimeError('offset_delta < 0.')

    dwell_time = args.dwell_time * U_('µs')
    if args.dwell_time < 0.:
        raise RuntimeError('dwell_time < 0.')

    outer_margin = args.outer_margin * U_('µm')
    if args.outer_margin < 0.:
        raise RuntimeError('outer_margin < 0.')

    oligomere_pattern = sample.Sample(f'Oligomere(n={n_monomers}, radius={mono_radius})')
    oligomere_site = oligomere_pattern.create_site(
        ([0., 0.], U_('µm')), ([.4, .4], U_('µm'))
    )

    beam = GaussBeam(Q_('2 nm'), Q_('2 pA'))

    mono_radius_with_gap = mono_radius + mono_gap / 2

    if n_monomers == 2:
        pass
    else:
        corner_polygon = Polygon.regular_ngon(
            n=n_monomers, radius=2*mono_radius_with_gap/(2.*np.sin(np.pi/n_monomers))
        )

        curves = make_oligomere_base_curves(corner_polygon, mono_radius_with_gap, not np.isclose(mono_gap, 0.))

        oligomere_site.add_pattern(Pattern(
            (curves['inner'], U_('µm')),
            Mill(dwell_time, inner_repeats, beam),
            raster.two_d.ContourParallel(
                offset_pitch=offset_delta,
                offset_direction='inwards',
                start_direction='inwards',
                scan_sequence=raster.ScanSequence.DOUBLE_SERPENTINE_SAME_PATH,
                line_style=raster.one_d.Curve(pitch=pitch_inner, scan_sequence=raster.ScanSequence.CONSECUTIVE),
                include_original_curve=False,
                optimize=True,
                smooth_radius=Q_('.5 nm')
            )
        ))

        oligomere_site.add_pattern(Pattern(
            (curves['outer'], U_('µm')),
            Mill(dwell_time, outer_repeats, beam),
            raster.two_d.ContourParallel(
                offset_pitch=offset_delta,
                offset_direction='outwards',
                start_direction='outwards',
                offset_distance=outer_margin,
                scan_sequence=raster.ScanSequence.DOUBLE_SERPENTINE_SAME_PATH,
                line_style=raster.one_d.Curve(pitch=pitch_outer, scan_sequence=raster.ScanSequence.CONSECUTIVE),
                include_original_curve=False,
                optimize=True,
                smooth_radius=Q_('0.5 nm')
            )
        ))

        if np.isclose(mono_gap, 0.):
            rasterized = RasterizedPoints.merged([rasterize(sep, pitch_outer.to('µm').m) for sep in curves['separators']])

            nominal_flux = beam.nominal_flux_per_spot_on_line(pitch_outer).to('ions / nm**2 / µs')
            dwell_points, hint, flux_matrix = optimze_rasterized_hybrid(rasterized.dwell_points.copy(), U_('µm'), beam, nominal_flux)

            oligomere_site.add_pattern(Pattern(
                (RasterizedPoints(dwell_points, is_closed=True), U_('µm')),
                Mill(dwell_time, mono_repeats),
                raster.zero_d.PreRasterized()
            ))

        else:
            raise NotImplementedError

        oligomere_pattern.export(SpotListBackend).save('pattern.txt')
        # oligomere_pattern.export(NPVETxt).save(
        #     f'tetramer_radius_{mono_radius}um_pitch_inner_{pitch_inner.m}um_pitch_outer_{pitch_outer.m}_inner_repeats_{inner_repeats}_outer_repeats_{outer_repeats}_mono_repeats_{mono_repeats}_offset_delta_{offset_delta.m}um_outer_margin_{outer_margin.m}um_partly_optimized' + '.txt')

# python use_cases/oligomere.py --monomers 4 --radius 0.05 --inner-repeats 2 --outer-repeats 2 --mono-repeats 50 --pitch-inner 0.0005 --pitch-outer 0.0005 --offset-delta 0.00025 --dwell-time 5 --outer-margin 0.04

if __name__ == '__main__':
    main()
