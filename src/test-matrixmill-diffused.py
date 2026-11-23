"""
Export Vasile dwell maps (.npz) to FEI stream files, one file per slice, with
error diffusion of the 25 ns time quantisation.

Why: the stream file stores dwell times as integer multiples of 25 ns. With
short visits (max 1 us per visit, out-in spiral -> every spot visited twice per
pass) the SIL apex only gets ~23 ns per visit, at r = 1 um ~40 ns. Plain rounding
then changes the dose by up to +-33 % along iso-dwell contours, and because the
same rounding is repeated in every pass and every slice this mills a dip at the
apex and concentric rings around it.

What this script does instead:
  1. Builds the out-in spiral exactly as raster_styles.two_d.Spiral does.
  2. Samples the dwell map at every spot with bilinear interpolation on the
     pixel-centred grid used by vasile (MatrixMill used floor() -> 1/2 pixel shift).
  3. Walks along the actual beam path (out and back in) and quantises every visit
     to 25 ns, carrying the rounding error to the next visit (1D error diffusion /
     sigma-delta). Neighbouring visits are <= 100 nm apart, the beam sigma is
     170 nm, so the beam averages the dithered dose back to the intended one.
     The cumulated dwell along any stretch of the path is exact to within 12.5 ns.
     Where < 25 ns per visit is wanted (apex) some visits become 0 ns; those spots
     are left out of the file, so every written dwell is >= 25 ns.
  4. Exports through the normal fibomat FEIStreamFile backend (same HFW/margin
     handling as test-matrixmill.py).

Run from fib-o-mat/src with the project venv, e.g.
    ../.venv/bin/python test-matrixmill-diffused.py
    ../.venv/bin/python test-matrixmill-diffused.py --out-dir ~/streamfiles --plot
"""
import argparse
import contextlib
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from fibomat import Sample, U_, Q_, DDDMill, shapes, raster_styles
from fibomat.default_backends.fei import FEIStreamFile
from fibomat.raster_styles import RasterStyle
from fibomat.rasterizedpattern import RasterizedPattern


# ----------------------------------------------------------------------------
# Settings (defaults = values of test-matrixmill.py)
# ----------------------------------------------------------------------------
NPZ_PATH = "/home/emma/Documents/FBI/fibomat-v.0.6/flatten-from-0-9-tangent-max-dwell-40-first-5-slices-halved-500nm-graben.npz"
FILE_PREFIX = "max40-500nm-graben-first-5-slices-halved-diffused"  # no '.' allowed (FEI backend cuts at the first '.')

FOV_UM = 30.0                 # field of view of the vasile grid (n * dx)
PATTERN_RADIUS_NM = 11500     # radius of the milled circle
POINT_PITCH_NM = 100          # pitch along the spiral
SPIRAL_PITCH_NM = 100         # distance between spiral arms
DIRECTION = "out-in"          # needed against redeposition; every spot is visited twice per pass

MAX_VISIT_US = 1.0            # longest single visit; sets n_rep
TIME_QUANTUM_US = 0.025       # stream file time unit (header "s16,25ns")
MARGIN = 0.76                 # same as before

BEAM_SIGMA_NM = 170           # only used for the dose check printed at the end


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
class PrecomputedPoints(RasterStyle):
    """Raster style that just hands precomputed (x, y, t) points to the backend."""

    def __init__(self, dwell_points_um_us: np.ndarray):
        self._points = np.asarray(dwell_points_um_us, dtype=float)

    @property
    def dimension(self) -> int:
        return 2

    def rasterize(self, dim_shape, mill, out_length_unit, out_time_unit) -> RasterizedPattern:
        if out_length_unit != U_("µm") or out_time_unit != U_("µs"):
            raise ValueError("PrecomputedPoints expects a backend working in µm and µs (FEIStreamFile does).")
        return RasterizedPattern(self._points.copy(), out_length_unit, out_time_unit)


def spiral_path_um(radius_nm, pitch_nm, spiral_pitch_nm, direction):
    """Beam path (x, y) in µm in visiting order, built by fibomat's Spiral raster style."""
    spiral = raster_styles.two_d.Spiral(
        pitch=pitch_nm * U_("nm"),
        spiral_pitch=spiral_pitch_nm * U_("nm"),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        direction=direction,
    )

    def unit_dwell(point: np.ndarray):
        return Q_(1.0, "microsecond")

    circ = shapes.Circle(r=radius_nm, center=(0, 0))
    rast = spiral.rasterize(circ * U_("nm"), DDDMill(unit_dwell, 1), U_("µm"), U_("µs"))
    return np.asarray(rast.dwell_points)[:, :2]


def sample_map(map_2d, xy_um, dx_um):
    """Bilinear lookup on the pixel-centred vasile grid: pixel i sits at (i - (n-1)/2) * dx."""
    n_rows, n_cols = map_2d.shape
    col = xy_um[:, 0] / dx_um + (n_cols - 1) / 2
    row = xy_um[:, 1] / dx_um + (n_rows - 1) / 2
    return map_coordinates(map_2d, [row, col], order=1, mode="constant", cval=0.0)


def diffuse_along_path(target_us, quantum_us):
    """Quantise visit by visit, carrying the rounding error to the next visit."""
    out = np.zeros_like(target_us)
    err = 0.0
    for k, value in enumerate(target_us):
        if value <= 0.0:          # no dose intended here: do not dump carried error into it
            err = 0.0
            continue
        want = value + err
        steps = max(round(want / quantum_us), 0)
        out[k] = steps * quantum_us
        err = want - out[k]
    return out


def blurred_dose(xy_um, dwell, n, dx_um, sigma_px):
    """Splat visits onto the vasile grid and blur with the beam (for checking only)."""
    col = xy_um[:, 0] / dx_um + (n - 1) / 2
    row = xy_um[:, 1] / dx_um + (n - 1) / 2
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    grid = np.zeros((n, n))
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                      (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        rr, cc = r0 + dr, c0 + dc
        ok = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
        np.add.at(grid, (rr[ok], cc[ok]), (dwell * w)[ok])
    return gaussian_filter(grid, sigma_px)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=NPZ_PATH)
    ap.add_argument("--out-dir", default=".", help="where the .str files are written")
    ap.add_argument("--prefix", default=FILE_PREFIX)
    ap.add_argument("--max-visit-us", type=float, default=MAX_VISIT_US)
    ap.add_argument("--no-diffusion", action="store_true", help="plain rounding (old behaviour), for comparison")
    ap.add_argument("--dry-run", action="store_true", help="compute and check, but write no files")
    ap.add_argument("--plot", action="store_true", help="plot the accumulated dose error at the end")
    args = ap.parse_args()

    if "." in args.prefix:
        raise ValueError("prefix must not contain '.', the FEI backend cuts the file name at the first '.'")

    dwell_maps = np.load(args.npz)["dwell_maps"]           # total dwell per pixel per slice [s]
    n = int(round(math.sqrt(dwell_maps.shape[1])))
    dx_um = FOV_UM / n
    visits_per_pass = 2 if DIRECTION in ("out-in", "in-out") else 1
    # dwell per spot must give the same dose per area as dwell per vasile pixel
    area_factor = (POINT_PITCH_NM * SPIRAL_PITCH_NM * 1e-6) / dx_um**2

    print("building spiral path ...")
    path_um = spiral_path_um(PATTERN_RADIUS_NM, POINT_PITCH_NM, SPIRAL_PITCH_NM, DIRECTION)
    print(f"{len(path_um)} visits per pass, {n}x{n} map, dx = {dx_um*1e3:.0f} nm, area factor {area_factor:.3f}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_px = BEAM_SIGMA_NM * 1e-3 / dx_um
    r_path = np.hypot(path_um[:, 0], path_um[:, 1])
    grid = (np.arange(n) - (n - 1) / 2) * dx_um
    r_grid = np.hypot(*np.meshgrid(grid, grid))
    check_region = r_grid < 4.5                                 # SIL top + flank
    acc_ideal = np.zeros((n, n))
    acc_real = np.zeros((n, n))

    for i, dwell_map in enumerate(dwell_maps):
        total_us = dwell_map.reshape(n, n) * 1e6 * area_factor    # wanted dwell per spot per slice [µs]
        n_rep = max(int(math.ceil(total_us.max() / (visits_per_pass * args.max_visit_us))), 1)
        per_visit_map = total_us / (visits_per_pass * n_rep)
        target = sample_map(per_visit_map, path_um, dx_um)       # wanted dwell per visit [µs], path order

        if args.no_diffusion:
            dwell = np.round(target / TIME_QUANTUM_US) * TIME_QUANTUM_US
        else:
            dwell = diffuse_along_path(target, TIME_QUANTUM_US)

        # --- check: beam-blurred dose of this slice, quantised vs. wanted ---
        d_ideal = blurred_dose(path_um, target, n, dx_um, sigma_px) * n_rep
        d_real = blurred_dose(path_um, dwell, n, dx_um, sigma_px) * n_rep
        acc_ideal += d_ideal
        acc_real += d_real
        rel = (d_real - d_ideal)[check_region] / d_ideal[check_region]
        apex = target[r_path < 0.15].mean() * 1e3
        print(f"slice {i:2d}: n_rep={n_rep:5d}  apex visit ~{apex:5.1f} ns  "
              f"0-ns visits dropped {int(np.sum(dwell == 0)):4d}  "
              f"total time error {(dwell.sum() / target.sum() - 1)*100:+.3f} %  "
              f"local dose error (r<4.5um) max {np.abs(rel).max()*100:4.1f} %")

        if args.dry_run:
            continue

        # Where less than 25 ns per visit is wanted (apex), the diffusion sets some visits to 0.
        # Leave those spots out of the file instead of writing a 0 dwell; that is the same dose.
        # The bounding box must not change, it sets HFW and centring of the stream file.
        keep = dwell > 0
        points = np.column_stack([path_um, dwell])
        if (points[keep, :2].min(0) != path_um.min(0)).any() or (points[keep, :2].max(0) != path_um.max(0)).any():
            print("  outermost spots have 0 dwell -> keeping all points to preserve HFW/centring")
            keep[:] = True
        points = points[keep]

        s = Sample()
        site = s.create_site(dim_position=(0, 0) * U_("µm"), dim_fov=(FOV_UM, FOV_UM) * U_("µm"))
        site.create_pattern(
            dim_shape=shapes.Circle(r=PATTERN_RADIUS_NM, center=(0, 0)) * U_("nm"),
            mill=DDDMill(lambda p: Q_(0.0, "microsecond"), 1),        # unused, points are precomputed
            raster_style=PrecomputedPoints(points),
        )
        exported = s.export(FEIStreamFile, n_rep=n_rep, margin=MARGIN)
        with contextlib.chdir(out_dir):     # relative name: the backend splits the path at the first '.'
            exported.save(f"{args.prefix}-{i}.str")

    rel_acc = (acc_real - acc_ideal) / np.where(acc_ideal > 0, acc_ideal, np.inf)
    print(f"\nall slices, local dose error r<4.5um: max {np.abs(rel_acc[check_region]).max()*100:.2f} %, "
          f"rms {rel_acc[check_region].std()*100:.2f} %")
    if not args.dry_run:
        print(f"stream files written to {out_dir}")

    if args.plot:
        import matplotlib.pyplot as plt
        ext = (grid[0], grid[-1], grid[0], grid[-1])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        lim = 0.6 * max(np.abs(rel_acc[check_region]).max() * 100, 0.1)
        im = ax[0].imshow(np.where(check_region, rel_acc * 100, np.nan), extent=ext, origin="lower",
                          cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[0].set(title="accumulated dose error [%] (+ = mills deeper)", xlabel="x [µm]", ylabel="y [µm]",
                  xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
        fig.colorbar(im, ax=ax[0])
        c = n // 2
        ax[1].plot(grid, rel_acc[c] * 100)
        ax[1].set(title="cut through centre", xlabel="x [µm]", ylabel="dose error [%]", xlim=(-4.5, 4.5))
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
