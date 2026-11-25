"""
Export Vasile dwell maps (.npz) to FEI stream files, one file per slice, with the
beam visiting exactly the pixel centres of the vasile grid along a spiral-ordered
path (out-in), and the 25 ns time quantisation diffused along that path.

Why pixel centres: the optimizer models one Gaussian spot per pixel centre with
the pixel's dwell time. Visiting exactly these points makes the stream file the
optimizer's model itself (up to the 25 ns rounding):
  - no interpolation of the dwell map (bilinear interpolation gave +0.3 nm extra depth),
  - no spiral / pixel-grid moiré (largest remaining difference with the Archimedean spiral),
  - no special centre spot (300x300 grid: the four pixels nearest the apex sit at ±50 nm),
  - the point set is symmetric, so the FEI export centres the pattern exactly (no 35 nm offset).
In simulation (all 24 slices) the result matches the npz simulation to 0.32 nm rms
within r < 2 µm, apex -0.09 nm (Archimedean spiral + centre fix: 0.67 nm rms).

Path: every pixel centre with r <= PATTERN_RADIUS_NM, sorted along an Archimedean
spiral whose arm spacing equals the pixel size (pixel assigned to arm
k = round(r/pitch - phi/2pi), sorted by k + phi/2pi). Outwards, then the same
points in reverse (out-in, against redeposition). Steps are 100 nm (5 % diagonal,
141 nm); a few longer jumps only at the outer rim, where the circle cuts the last arm.

Quantisation: with out-in each pixel is visited twice per pass. The wanted time of
both visits together (2 t) is rounded to 25 ns steps, the rounding error is carried
to the next pixel along the outward path, and the steps are split between the outward
and the inward visit. Visits that end up with 0 ns are left out of the file.

The spot pitch is the optimizer's pixel size (100 nm). A different pitch needs a new
optimisation on that grid.

Run from fib-o-mat/src with the project venv, e.g.
    ../.venv/bin/python test-matrixmill-pixelspiral.py --out-dir ~/streamfiles
    ../.venv/bin/python test-matrixmill-pixelspiral.py --dry-run --plot
"""
import argparse
import contextlib
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from fibomat import Sample, U_, Q_, DDDMill, shapes
from fibomat.default_backends.fei import FEIStreamFile
from fibomat.raster_styles import RasterStyle
from fibomat.rasterizedpattern import RasterizedPattern


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
NPZ_PATH = "/home/emma/Documents/FBI/fibomat-v.0.6/flatten-from-0-9-tangent-max-dwell-40-first-5-slices-halved-500nm-graben.npz"
FILE_PREFIX = "max40-500nm-graben-first-5-slices-halved-pixelspiral"  # no '.' allowed (FEI backend cuts at the first '.')

FOV_UM = 30.0                 # field of view of the vasile grid (n * dx)
PATTERN_RADIUS_NM = 11500     # pixels with centre radius <= this are milled

MAX_VISIT_US = 1.0            # longest single visit; sets n_rep
TIME_QUANTUM_US = 0.025       # stream file time unit (header "s16,25ns")
MARGIN = 0.76                 # same as before
VISITS_PER_PASS = 2           # out-in: every pixel is visited twice per pass

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


def pixel_spiral(n, dx_um, radius_um):
    """Pixel centres inside radius_um in Archimedean-spiral order (outwards).

    Returns (xy_um [N, 2], flat_index [N]) with flat_index into the row-major (y, x) map.
    Pixel i sits at (i - (n-1)/2) * dx, as in vasile's get_target_from_mill.
    """
    axis = (np.arange(n) - (n - 1) / 2) * dx_um
    X, Y = np.meshgrid(axis, axis)                       # [row = y, col = x]
    inside = np.hypot(X, Y) <= radius_um
    x, y = X[inside], Y[inside]
    flat = np.flatnonzero(inside)
    r = np.hypot(x, y)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    arm = np.round(r / dx_um - phi / (2 * np.pi))       # which arm of a spiral with pitch dx
    order = np.argsort(arm + phi / (2 * np.pi), kind="stable")
    return np.column_stack([x[order], y[order]]), flat[order]


def paired_diffusion(per_visit_us, quantum_us):
    """Round the two visits of each pixel together, carrying the error along the path.

    per_visit_us: wanted time per visit, outward path order.
    Returns (outward_us, inward_us), both in outward path order.
    """
    steps = np.zeros(len(per_visit_us), dtype=np.int64)
    err = 0.0
    for k, value in enumerate(per_visit_us):
        if value <= 0.0:          # no dose intended here: do not dump carried error into it
            err = 0.0
            continue
        want = 2.0 * value + err
        steps[k] = max(round(want / quantum_us), 0)
        err = want - steps[k] * quantum_us
    out_steps = steps // 2
    in_steps = steps - out_steps
    return out_steps * quantum_us, in_steps * quantum_us


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=NPZ_PATH)
    ap.add_argument("--out-dir", default=".", help="where the .str files are written")
    ap.add_argument("--prefix", default=FILE_PREFIX)
    ap.add_argument("--max-visit-us", type=float, default=MAX_VISIT_US)
    ap.add_argument("--dry-run", action="store_true", help="compute and check, but write no files")
    ap.add_argument("--plot", action="store_true", help="plot the path and the accumulated dose error")
    args = ap.parse_args()

    if "." in args.prefix:
        raise ValueError("prefix must not contain '.', the FEI backend cuts the file name at the first '.'")

    dwell_maps = np.load(args.npz)["dwell_maps"]          # total dwell per pixel per slice [s]
    n = int(round(math.sqrt(dwell_maps.shape[1])))
    dx_um = FOV_UM / n

    path_out, flat = pixel_spiral(n, dx_um, PATTERN_RADIUS_NM * 1e-3)
    path_in = path_out[::-1]
    path_all = np.concatenate([path_out, path_in])
    steps = np.hypot(*np.diff(path_out, axis=0).T) * 1e3
    bbox_centre = (path_all.min(0) + path_all.max(0)) / 2 * 1e3
    print(f"{len(path_out)} pixels on the path ({len(path_all)} visits per pass), {n}x{n} map, dx = {dx_um*1e3:.0f} nm")
    print(f"steps: median {np.median(steps):.0f} nm, 95 % <= {np.percentile(steps, 95):.0f} nm, "
          f"{np.sum(steps > 300)} jumps > 300 nm (outer rim, r >= {np.hypot(*path_out[1:][steps > 300].T).min():.2f} µm)"
          if np.any(steps > 300) else "")
    print(f"bounding box centre of the points: ({bbox_centre[0]:+.2f}, {bbox_centre[1]:+.2f}) nm")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_px = BEAM_SIGMA_NM * 1e-3 / dx_um
    axis = (np.arange(n) - (n - 1) / 2) * dx_um
    R = np.hypot(*np.meshgrid(axis, axis))
    check_region = R < 4.5
    r_path = np.hypot(path_out[:, 0], path_out[:, 1])
    acc_wanted = np.zeros(n * n)
    acc_real = np.zeros(n * n)

    for i, dwell_map in enumerate(dwell_maps):
        total_us = dwell_map.ravel() * 1e6                            # wanted dwell per pixel per slice [µs]
        n_rep = max(int(math.ceil(total_us[flat].max() / (VISITS_PER_PASS * args.max_visit_us))), 1)
        per_visit = total_us[flat] / (VISITS_PER_PASS * n_rep)        # outward path order
        t_out, t_in = paired_diffusion(per_visit, TIME_QUANTUM_US)
        dwell = np.concatenate([t_out, t_in[::-1]])                    # visit order: out, then back in

        # --- check: beam-blurred dose of this slice, as written vs. wanted ---
        wanted = np.zeros(n * n)
        real = np.zeros(n * n)
        wanted[flat] = total_us[flat]
        real[flat] = (t_out + t_in) * n_rep
        acc_wanted += wanted
        acc_real += real
        bw = gaussian_filter(wanted.reshape(n, n), sigma_px)
        br = gaussian_filter(real.reshape(n, n), sigma_px)
        rel = (br - bw)[check_region] / bw[check_region]
        apex = per_visit[r_path < 0.1].mean() * 1e3
        print(f"slice {i:2d}: n_rep={n_rep:5d}  apex visit ~{apex:5.1f} ns  "
              f"0-ns visits dropped {int(np.sum(dwell == 0)):5d}  "
              f"total time error {(real.sum() / wanted.sum() - 1) * 100:+.3f} %  "
              f"local dose error (r<4.5um) max {np.abs(rel).max() * 100:4.1f} %")

        if args.dry_run:
            continue

        # Leave visits with 0 ns out of the file. The bounding box must not change, it sets HFW and centring.
        keep = dwell > 0
        if (path_all[keep].min(0) != path_all.min(0)).any() or (path_all[keep].max(0) != path_all.max(0)).any():
            print("  outermost points have 0 dwell -> keeping all points to preserve HFW/centring")
            keep[:] = True
        points = np.column_stack([path_all, dwell])[keep]

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

    bw = gaussian_filter(acc_wanted.reshape(n, n), sigma_px)
    br = gaussian_filter(acc_real.reshape(n, n), sigma_px)
    rel_acc = (br - bw) / np.where(bw > 0, bw, np.inf)
    print(f"\nall slices, local dose error r<4.5um: max {np.abs(rel_acc[check_region]).max() * 100:.2f} %, "
          f"rms {rel_acc[check_region].std() * 100:.2f} %")
    if not args.dry_run:
        print(f"stream files written to {out_dir}")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        m = r_path < 0.6
        ax[0].plot(path_out[m, 0], path_out[m, 1], "-o", ms=3)
        ax[0].plot(path_out[0, 0], path_out[0, 1], "rs", label="start")
        ax[0].set(aspect="equal", title="path near the apex (outward run)", xlabel="x [µm]", ylabel="y [µm]")
        ax[0].legend()
        ext = (axis[0], axis[-1], axis[0], axis[-1])
        lim = 0.6 * max(np.abs(rel_acc[check_region]).max() * 100, 0.1)
        im = ax[1].imshow(np.where(check_region, rel_acc * 100, np.nan), extent=ext, origin="lower",
                          cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[1].set(title="accumulated dose error [%] (+ = mills deeper)", xlabel="x [µm]", ylabel="y [µm]",
                  xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
        fig.colorbar(im, ax=ax[1])
        c = n // 2
        ax[2].plot(axis, rel_acc[c] * 100)
        ax[2].set(title="cut through centre", xlabel="x [µm]", ylabel="dose error [%]", xlim=(-4.5, 4.5))
        ax[2].grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
