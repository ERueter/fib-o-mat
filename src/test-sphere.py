from fibomat import Sample, U_, Mill, Q_, DDDMill, SILMill, MatrixMill
from fibomat import shapes, raster_styles
from fibomat import Vector
from fibomat.default_backends.fei import FEIStreamFile, SpotListBackend
import numpy as np
from fibomat.units import QuantityType
from fibomat.shapes import ParametricCurve
from fibomat.optimize import vasile_with_fft_structured as vas
from fibomat.calibrate import calibrate
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d


s = Sample()

config = vas.ProcessConfig(n=300,dx=100e-9, dy=100e-9,use_numpy_grad=True, Y0=0.95, p=-2.76, q=-0.68, h=9.6e28, f_xy=7.2e22)#n=1200 #mit dem katja sputteryield werden Y,p,q nicht benutzt# q in exp, p in cos # 0.26 mm per second of millng    material_scale=0.5575e-3
def postprocess(D_vec, t_clip, C_dot, CT_dot, n):
    return t_clip

# TODO Frage an Katja: Die sil mit maxdwell-time 10 zu machen und dann einfach sehr oft zu millen müsste eigentlich in falscher shape resultieren?
#silmill = SILMill(radius_sil=3953*U_('nm'),radius=7370*U_('nm'), min_dwell_time=0.1)  # arbeitet in µs
silmill = SILMill(radius_sil=5000*U_('nm'),radius=10000*U_('nm'), min_dwell_time=0.1)  # arbeitet in µs

# TODO silmill is larger than fov in vasile rn

radius_sil=silmill._radius_sil
radius=silmill._radius
min_dwell_time=0.025
max_dwell_time = 10

flatten_from = 0.9


def dwell_func(point: np.ndarray) -> QuantityType:
    x, y = point[0], point[1]
    dist_sq = x * x + y * y
    dist = np.sqrt(dist_sq)

    # Shift apex downward by 10% of SIL radius
    z_shift = 500 # 50 nm

    if dist < radius_sil:
        # Original spherical cap term, shifted downward
        sag = np.sqrt(radius_sil**2 - dist_sq)
        depth = max_dwell_time - (max_dwell_time / radius_sil) * (sag - z_shift)

    elif dist < radius:
        # Keep outer taper continuous from SIL boundary
        edge_depth = max_dwell_time - (max_dwell_time / radius_sil) * (
            np.sqrt(radius_sil**2 - radius_sil**2) - z_shift
        )
        depth = edge_depth * (1 - (dist - radius_sil) / (radius - radius_sil))

    else:
        return Q_(0, "microsecond")

    return Q_(max(depth, min_dwell_time), "microsecond")


def dwell_func(point: np.ndarray) -> QuantityType:
    x, y = point[0], point[1]
    r = np.sqrt(x * x + y * y)

    R_sil = 5000.0       # nm
    R_outer = 12000.0    # nm
    R_blend = 1000.0     # nm

    if r > R_outer:
        return Q_(0, "microsecond")

    # ------------------------------------------------------------
    # 1. Exakte Kugelhälfte
    # ------------------------------------------------------------
    if r <= R_sil:
        depth = R_sil - np.sqrt(
            max(R_sil**2 - r**2, 0.0)
        )

    else:
        # --------------------------------------------------------
        # 2. Blend-Kreis
        # --------------------------------------------------------
        x_center = R_sil + R_blend
        z_center = R_sil

        # --------------------------------------------------------
        # 3. Tangentialpunkt des Kreises mit der Auslaufgeraden
        #
        # Die Gerade muss durch
        #     P_outer = (R_outer, 0)
        # gehen und den Kreis tangential berühren.
        #
        # Bedingung:
        #   (P_t - C) · (P_t - P_outer) = 0
        #
        # Für den hier verwendeten oberen Kreisabschnitt ergibt sich
        # der relevante Tangentialpunkt analytisch.
        # --------------------------------------------------------
        dx_outer = R_outer - x_center
        dz_outer = -z_center

        d2 = dx_outer**2 + dz_outer**2
        d = np.sqrt(d2)

        # Winkel vom Kreismittelpunkt zum äußeren Punkt
        theta = np.arctan2(dz_outer, dx_outer)

        # Winkel zwischen C->P_outer und C->P_t
        alpha = np.arccos(R_blend / d)

        # Relevanter Tangentialpunkt auf dem oberen Kreis
        theta_t = theta + alpha

        r_tangent = x_center + R_blend * np.cos(theta_t)
        z_tangent = z_center + R_blend * np.sin(theta_t)

        if r <= r_tangent:
            # Kreisstück
            dx = r - x_center

            depth = z_center + np.sqrt(
                max(R_blend**2 - dx**2, 0.0)
            )

        else:
            # ----------------------------------------------------
            # 3. Linearer Auslauf
            #
            # Gerade vom Tangentialpunkt bis
            # (R_outer, 0).
            # ----------------------------------------------------
            m_line = (
                0.0 - z_tangent
            ) / (
                R_outer - r_tangent
            )

            depth = z_tangent + m_line * (
                r - r_tangent
            )

    dwell = max_dwell_time * depth / R_sil

    return Q_(
        max(dwell, min_dwell_time),
        "microsecond"
    )


R_sil = 5000.0       # nm

R_outer = 12000.0    # nm
profile_blur_sigma = 300.0  # nm
profile_grid_step = 10.0    # nm


def _raw_profile_depth(r: float) -> float:
    r0 = flatten_from * R_sil
    depth0 = R_sil - np.sqrt(R_sil**2 - r0**2)
    sphere_slope_at_r0 = r0 / np.sqrt(R_sil**2 - r0**2)
    depth_at_sil_edge = depth0 + sphere_slope_at_r0 * (R_sil - r0)

    if r <= r0:
        return R_sil - np.sqrt(max(R_sil**2 - r**2, 0.0))

    if r <= R_sil:
        t = (r - r0) / (R_sil - r0)
        blend = 6 * t**5 - 15 * t**4 + 10 * t**3
        sphere_depth = R_sil - np.sqrt(max(R_sil**2 - r**2, 0.0))
        tangent_depth = depth0 + sphere_slope_at_r0 * (r - r0)
        return (1 - blend) * sphere_depth + blend * tangent_depth

    length = R_outer - R_sil
    t = (r - R_sil) / length
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (
        h00 * depth_at_sil_edge
        + h10 * length * sphere_slope_at_r0
        + h01 * 0.0
        + h11 * length * 0.0
    )


_profile_r = np.arange(
    0.0, R_outer + profile_grid_step, profile_grid_step
)
_profile_depth_raw = np.array(
    [_raw_profile_depth(r) for r in _profile_r]
)
_profile_depth_blurred = gaussian_filter1d(
    _profile_depth_raw,
    sigma=profile_blur_sigma / profile_grid_step,
    mode="constant",
    cval=0.0,
)
_profile_depth_blurred[-1] = 0.0


def dwell_func(point: np.ndarray) -> QuantityType:
    x, y = point[0], point[1]
    r = np.sqrt(x * x + y * y)

    if r >= R_outer:
        return Q_(0, "microsecond")

    r0 = flatten_from * R_sil
    if r <= r0:
        depth = R_sil - np.sqrt(max(R_sil**2 - r**2, 0.0))
    else:
        raw_depth = _raw_profile_depth(r)
        blurred_depth = np.interp(r, _profile_r, _profile_depth_blurred)

        # Fade in the blur over three sigma so the exact sphere joins smoothly.
        blend_t = min((r - r0) / (3 * profile_blur_sigma), 1.0)
        blend = 6 * blend_t**5 - 15 * blend_t**4 + 10 * blend_t**3
        depth = (1 - blend) * raw_depth + blend * blurred_depth

    dwell = min_dwell_time + (
        (max_dwell_time - min_dwell_time) * max(depth, 0.0) / R_sil
    )
    return Q_(dwell, "microsecond")


rounding_length_func4 = 2000#2000.0  # nm; radial length of the parabolic transition
rounding_depth_reduction_func4 = 800.0
rounding_adjustment = -1000  # controlled flattening of the parabola

r0_func4 = flatten_from * R_sil
depth0_func4 = R_sil - np.sqrt(R_sil**2 - r0_func4**2)
tangent_slope_func4 = r0_func4 / np.sqrt(R_sil**2 - r0_func4**2)
tangent_end_r_func4 = (
    r0_func4 + (R_sil - depth0_func4) / tangent_slope_func4
)
linear_slope_func4 = -1.0  # 45 degrees: 1 nm less depth per 1 nm radial distance
round_end_r_func4 = tangent_end_r_func4 + rounding_length_func4
round_end_depth_func4 = (
    R_sil
    + (tangent_slope_func4 + linear_slope_func4)
    * rounding_length_func4 / 2
)
profile_end_r_func4 = (
    round_end_r_func4 - round_end_depth_func4 / linear_slope_func4
)




def _raw_profile_depth_func4(r: float) -> float:
    if rounding_length_func4 <= 0:
        raise ValueError(
            "rounding_length_func4 must be positive"
        )

    parabola_curvature = (
        linear_slope_func4 - tangent_slope_func4
    ) / (2 * rounding_length_func4)

    if r <= r0_func4:
        return R_sil - np.sqrt(max(R_sil**2 - r**2, 0.0))

    if r <= tangent_end_r_func4:
        return depth0_func4 + tangent_slope_func4 * (r - r0_func4)

    if r <= round_end_r_func4:

        distance_from_tangent_end = r - tangent_end_r_func4

        return (
            R_sil
            + tangent_slope_func4 * distance_from_tangent_end
            + parabola_curvature * distance_from_tangent_end**2
        )

    return max(
        0.0,
        round_end_depth_func4
        + linear_slope_func4 * (r - round_end_r_func4),
    )

############################## TEST ##############################

# ------------------------------------------------------------
# Parameter
# ------------------------------------------------------------

rounding_start_r = tangent_end_r_func4
rounding_end_r = round_end_r_func4

# Gewünschte Tiefe der horizontalen Verbindung
flat_depth = 5500.0  # nm

# Gauß-Glättung
profile_blur_sigma = 500.0  # nm
profile_grid_step = 10.0    # nm


# ------------------------------------------------------------
# Rohprofil erzeugen
# ------------------------------------------------------------

def _raw_profile_depth_func4(r: float) -> float:

    if r <= r0_func4:
        return R_sil - np.sqrt(
            max(R_sil**2 - r**2, 0.0)
        )

    if r <= tangent_end_r_func4:
        return (
            depth0_func4
            + tangent_slope_func4 * (r - r0_func4)
        )

    if r <= rounding_start_r:
        return (
            R_sil
            + tangent_slope_func4
            * (r - tangent_end_r_func4)
        )

    if r <= rounding_end_r:
        return flat_depth

    if r <= profile_end_r_func4:
        return (
            flat_depth
            + linear_slope_func4
            * (r - rounding_end_r)
        )

    return 0.0


# ------------------------------------------------------------
# Profil diskret berechnen
# ------------------------------------------------------------

_profile_r = np.arange(
    0.0,
    profile_end_r_func4 + 2 * profile_grid_step,
    profile_grid_step
)

_profile_depth_raw = np.array([
    _raw_profile_depth_func4(r)
    for r in _profile_r
])


# ------------------------------------------------------------
# Gauß-Glättung
# ------------------------------------------------------------

_profile_depth_blurred = gaussian_filter1d(
    _profile_depth_raw,
    sigma=profile_blur_sigma / profile_grid_step,
    mode="nearest"
)


# Kugel exakt erhalten
sphere_mask = _profile_r <= r0_func4

_profile_depth_blurred[sphere_mask] = (
    _profile_depth_raw[sphere_mask]
)


# ------------------------------------------------------------
# Finale Profilfunktion
# ------------------------------------------------------------

def _raw_profile_depth_func4(r: float) -> float:

    if r <= r0_func4:
        return R_sil - np.sqrt(
            max(R_sil**2 - r**2, 0.0)
        )

    return np.interp(
        r,
        _profile_r,
        _profile_depth_blurred,
        right=0.0
    )

########################## end of test #######################################
def dwell_func(point: np.ndarray) -> QuantityType:
    x, y = point[0], point[1]
    r = np.sqrt(x * x + y * y)

    if r >= profile_end_r_func4:
        return Q_(min_dwell_time, "microsecond")

    depth = _raw_profile_depth_func4(r)

    dwell = min_dwell_time + (
        (max_dwell_time - min_dwell_time) * max(depth, 0.0) / R_sil
    )
    return Q_(max(dwell, min_dwell_time), "microsecond")#Q_(dwell, "microsecond")



import matplotlib.pyplot as plt

# x-Achse von -R_outer bis +R_outer
x_values = np.linspace(-15000, 15000, 2000)

dwell_values = np.array([
    dwell_func(np.array([x, 0.0])).magnitude
    for x in x_values
])
depth_values = (
    np.maximum(dwell_values - min_dwell_time, 0.0)
    * radius_sil
    / (max_dwell_time - min_dwell_time)
)

plt.figure(figsize=(9, 5))
plt.plot(x_values / 1000, depth_values / 1000)
hemisphere_x_nm = np.linspace(-radius_sil, radius_sil, 1001)
hemisphere_depth_nm = radius_sil - np.sqrt(
    np.maximum(radius_sil**2 - hemisphere_x_nm**2, 0.0)
)
plt.plot(
    hemisphere_x_nm / 1000,
    hemisphere_depth_nm / 1000,
    "--",
    color="black",
    alpha=0.7,
    label="Exakte Kugel",
)
plt.axvline(-flatten_from * radius_sil / 1000, color="gray", linestyle="--", alpha=0.6)
plt.axvline(flatten_from * radius_sil / 1000, color="gray", linestyle="--", alpha=0.6)

plt.xlabel("x [µm]")
plt.ylabel("Abtragstiefe [µm]")
plt.title("Querschnitt der dwell_func entlang der x-Achse")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


mill = DDDMill(dwell_func, 1)

#mill = silmill

spiral_style = raster_styles.two_d.Spiral(pitch=20 * U_('nm'),spiral_pitch=20 * U_('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE, direction="out-in")

#a, repeats_for_depth = calibrate.calibrate(rasterstyle=spiral_style)

circ = shapes.Circle(r=13583, center=(0,0))


target_depth = 4.18*U_("µm")#6*U_('µm') # µm


#repeats = repeats_for_depth(target_depth)

#print("repeats = " + str(repeats)) # 758 repeats = a mill with max-dwelltime 10 µs has to mill 758 times in total to reach depth 7.3 µm


print("berechne target")
Z_target, dx = vas.get_target_from_mill(
    mill=mill,
    resolution=config.n,            # choose desired resolution of the target
    fov=30 * U_('µm'),         # your site FOV
    unit=U_("nm"),                 # eigentlich muss hier das site-unit stehen? #mill receives µm coordinates
    verbose=True
)

# the Z_target from vas now basically just has the dwell-time at each point - so technically it is in µs, not in µm. 
# we now basically use the mill to (by abuse of notation) encode a depth by just scaling it to the target depth.
# so the mill just stored the shape for us, not the depth (the user has to know themself what depth they want)

max_Z = np.max(Z_target)

target_depth = (
    max_Z
    * R_sil
    / (1000 * (max_dwell_time - min_dwell_time))
) * U_("µm")

scale = target_depth.magnitude/max_Z
Z_target = Z_target*scale # now Z_target has the targeted depth and is in units of target depth (though dimensionless stored for vas)
Z_target *= 1e-6 # convert to [m] for vasile

if False:  # I believe this was bullshit?
    # Skaliere Z_target so, dass der tiefste Punkt 7.3 µm entspricht
    max_Z = np.max(Z_target)
    r = a / 10 * 1e-6  # experimental milling rate in m/s (µm/µs * 1e-6)
    original_max_depth = r * max_Z  # original max depth in m
    scale = target_depth.magnitude / original_max_depth 
    Z_target = Z_target * scale

    Z_target *= 1e-6

    Z_target = config.f_xy / config.h * Z_target  # Z umrechnen von der ZEIT zu der TIEFE

if True:
    import matplotlib.pyplot as plt
    import numpy as np

    resolution = Z_target.shape[0]
    grid_nm = (np.arange(resolution) - resolution / 2 + 0.5) * dx

    lower_y_idx = np.searchsorted(grid_nm, 0, side="right") - 1
    upper_y_idx = lower_y_idx + 1

    y_fraction = -grid_nm[lower_y_idx] / (
        grid_nm[upper_y_idx] - grid_nm[lower_y_idx]
    )

    target_x_section = (
        (1 - y_fraction) * Z_target[lower_y_idx, :]
        + y_fraction * Z_target[upper_y_idx, :]
    )

    x_axis_um = grid_nm / 1000

    # Perfekte Halbkugel
    R_sil = 5000.0  # nm
    hemisphere_x_nm = np.linspace(-R_sil, R_sil, 1001)
    hemisphere_r_nm = np.abs(hemisphere_x_nm)
    hemisphere_depth_nm = R_sil - np.sqrt(
        np.maximum(R_sil**2 - hemisphere_r_nm**2, 0.0)
    )
    hemisphere_x_um = hemisphere_x_nm / 1000
    hemisphere_depth_um = hemisphere_depth_nm / 1000

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        x_axis_um,
        target_x_section * 1e6,
        label="Target depth",
        linewidth=2
    )

    plt.plot(
        hemisphere_x_um,
        hemisphere_depth_um,
        "--",
        label="Perfect hemisphere (R = 5 µm)",
        linewidth=2
    )

    plt.xlabel("x [µm]")
    plt.ylabel("Target depth [µm]")
    plt.title("Z_target cross-section at y = 0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()



print("plot kommt")
import matplotlib.pyplot as plt
resolution = Z_target.shape[0]
grid_nm = (np.arange(resolution) - resolution / 2 + 0.5) * dx
lower_y_idx = np.searchsorted(grid_nm, 0, side="right") - 1
upper_y_idx = lower_y_idx + 1
y_fraction = -grid_nm[lower_y_idx] / (grid_nm[upper_y_idx] - grid_nm[lower_y_idx])
target_x_section = (
    (1 - y_fraction) * Z_target[lower_y_idx, :]
    + y_fraction * Z_target[upper_y_idx, :]
)
x_axis_um = grid_nm / 1000

plt.figure(figsize=(8, 5))
plt.plot(x_axis_um, target_x_section * 1e6)
plt.xlabel("x [µm]")
plt.ylabel("Target depth [µm]")
plt.title("Z_target cross-section at y = 0")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

grid_edges_um = (
    np.arange(resolution + 1) - resolution / 2
) * dx / 1000
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(
    Z_target * 1e6,
    extent=(grid_edges_um[0], grid_edges_um[-1], grid_edges_um[0], grid_edges_um[-1]),
    origin="lower",
    cmap="viridis",
    aspect="equal",
)
plt.xlabel("x [µm]")
plt.ylabel("y [µm]")
plt.title("Z_target heatmap")
plt.colorbar(heatmap, label="Target depth [µm]")
plt.tight_layout()
plt.show()
#print(f"Z_target skaliert mit Faktor {scale:.4f}, max Z_target: {np.max(Z_target):.2f} µs")

dz = 250e-9 #7500 nm / 20 slices = 375e-9 250e-9 # 7500 nm /30 slices = 250 nm per slice #0.5e-7  # tiefe pro Slice in m, 
Z_blurred = vas.preprocess_Z(Z_target, config, verbose = False)
Z_final, dwell_maps, surface_history = vas.process_full_target(
    Z_target=Z_blurred,
    dz=dz,
    config=config,
    postprocess=postprocess,
    verbose=False,
    slice_mode="envelope",
    record_surface_history=True
)

vas.plot_surface_history(surface_history, Z_blurred, config)
vas.evaluate_accuracy(Z_blurred, Z_final, dwell_maps, config)

# save in current directory with filename simulation_results_sine
np.savez("smoothed-sil-flatten_from-0-8.npz",
         Z_final=Z_final,
         dwell_maps=dwell_maps,
         Z_target=Z_blurred)

exit()

for i, dwell_map in enumerate(dwell_maps):
    print(f"Layer {i}: max_d = {np.max(dwell_map):.2e} s, shape = {dwell_map.shape}")
    max_d = np.max(dwell_map)
    if max_d > 0:
        scale = 10e-6 / max_d  # scale to max 10 µs
        dwell_map_scaled = dwell_map * scale
        n = int(np.sqrt(dwell_map_scaled.size))
        dwell_map_scaled = dwell_map_scaled.reshape((n, n))
        n_rep = int(np.ceil(max_d / 10e-6))
        
        """
        # Plot original and scaled dwell maps
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(dwell_map.reshape((n, n)), cmap='viridis', origin='lower')
        plt.colorbar(label='Dwell time (s)')
        plt.title(f'Layer {i}: Original Dwell Map (max: {max_d:.2e} s)')
        
        plt.subplot(1, 2, 2)
        plt.imshow(dwell_map_scaled, cmap='viridis', origin='lower')
        plt.colorbar(label='Dwell time (s)')
        plt.title(f'Layer {i}: Scaled Dwell Map (max: {np.max(dwell_map_scaled):.2e} s)')
        plt.show()
        """
        
    else:
        print("bin im else-fall!")
        dwell_map_scaled = np.zeros_like(dwell_map)  # ensure zeros
        n = int(np.sqrt(dwell_map_scaled.size))
        dwell_map_scaled = dwell_map_scaled.reshape((n, n))
        n_rep = 0  # skip if zero
    
    origin = (-10, -10)  # TODO check how the origin has to look like
    dwell_map_scaled *= 1e6  # convert to µs for MatrixMill TODO check this
    layer_mill = MatrixMill(dwell_map_scaled, dx=0.05, origin=origin)
    
    """
    # Sample from layer_mill for visualization
    x = np.linspace(origin[0], origin[0] + n * 0.05, 100)
    y = np.linspace(origin[1], origin[1] + n * 0.05, 100)
    X, Y = np.meshgrid(x, y)
    dwell_samples = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dwell_samples[i, j] = layer_mill.dwell_time(np.array([X[i, j], Y[i, j]])).to('second').magnitude
    
    plt.figure(figsize=(6, 5))
    plt.imshow(dwell_samples, extent=[origin[0], origin[0] + n * 0.05, origin[1], origin[1] + n * 0.05], origin='lower', cmap='viridis')
    plt.colorbar(label='Sampled Dwell time (s)')
    plt.title(f'Layer {i}: Sampled from layer_mill')
    plt.show()
    """

    
    # Create a new site for each layer
    layer_site = s.create_site(
        dim_position=(0, 0) * U_('µm'), dim_fov=(20, 20) * U_('µm')
    )

    circ = shapes.Circle(r=7370, center=(0,0))  # warum auch immer es crasht, wenn derselbe Kreis mehrfach benutzt wird???
    
    layer_site.create_pattern(
        dim_shape=circ * U_('nm'),
        mill=layer_mill,
        raster_style=spiral_style
    )
    
    # TODO hier muss das sample exported werden, sites können nicht exportiert werden. einfach sites löschen oder so I guess.
    #s.plot(rasterize_pitch=Q_('0.01 µm'), plot_rasterized=True)
    # plot von erster map sieht normal aus, das zweite ist nur ein Punkt!!!
    exported = s.export(FEIStreamFile, n_rep=n_rep, margin=0.76) 
    exported.save(f'sil-optimized-layer-{i}.str')
    print(f"Layer {i}: max dwell {max_d:.2f} µs, scale {scale:.4f}, n_rep {n_rep}")
    s.empty_sites()

raise Exception("end of vasile test")

print("Before export")
exported = s.export(FEIStreamFile, n_rep=repeats, margin=0.76) 
print("Before save")
exported.save('kalibriertes-file.str')
print("File saved successfully")











raise Exception("end of test")


#---------------------------------------------------- Vasile Test -------------------------------------------------------
# TODO: im Versuch auf dem Mikroskop waren die repeats des files am Ende 684. Damit die hier mit für die Wunschtiefe beachtet werden, müssen sie aber in der
# mill stehen, nicht nur im File...

# TODO: Vasile wie folgt einbauen: Die Mill muss optimiert werden!
# Dafür die vom User als Funktion übergebene Mill auf einer gewissen Genauigkeit samplen, sodass man die Tiefen-Matrix wie sonst immer für Vasile benutzt kriegt.
# Das dann wie gekannt optimieren, danach eine einfache "aus der Matrix auslesen"-Funktion als neue Mill setzen.
# TODO: units angucken. aktuell ist die Gesamtzeit am Ende bis zu 5e+9, das wirkt sehr hoch? andererseits, 5ns... im streamfile sind es 2,5 µs... aber das dann mit repeats, aber die sind hier 
# auch noch nicht drin? 
config = vas.ProcessConfig(use_numpy_grad=False)
def postprocess(D_vec, t_clip, C_dot, CT_dot, n):
    return t_clip


# TODO Z_target ist jetzt einfach in den units der mill, das nochmal klären
Z_target, dx = vas.get_target_from_mill(
    mill=mill,
    resolution=400,            # choose desired resolution of the target
    fov=20 * U_('µm'),         # your site FOV
    unit=U_("nm"),                 # eigentlich muss hier das site-unit stehen? #mill receives µm coordinates
    verbose=True
)


import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.imshow(Z_target, cmap='viridis', origin='lower', interpolation='nearest')
plt.colorbar(label="Dwell time [µs]")
plt.title("Generated SIL Target (from SILMill)")
plt.show()


Z_target *= 1e-6 # mill gibt in µs aus, wir brauchen es in s, um dann in der nächsten Zeile die Tiefe auszurechnen TODO weniger scuffed machen 
Z_target = config.f_xy / config.h * Z_target  # Z umrechnen von der ZEIT, die an einem Pixel verbracht werden soll, zu der TIEFE, welche die Shape da haben soll.
# andererseits ist das irgendwie doppelt? dann muss man für mill erst tiefe -> zeit rechnen und dann hier wieder zurück... aber evt. beste Lösung so.
#Z_target *= 1e-2 # umrechnen von cm in m
#Z_target *= 1e3  # TODO rausfinden, wo du Faktor verloren hast?
# 2,1 nA, 400 nm spotdurchmesser (halbwertsbreite gauss), 

# -----------------------------------------------------
# Plot result
# -----------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.imshow(Z_target, cmap='viridis', origin='lower', interpolation='nearest')
plt.colorbar(label="Target Depth [m]")
plt.title("Generated SIL Target (from SILMill)")
plt.show()


# -> soll 2 nm tief werden, keine Ahnung, ob das Sinn ergibt?


dz = 1e-6  # tiefe pro Slice in m
Z_blurred = vas.preprocess_Z(Z_target, config, verbose = False)
Z_final, dwell_maps = vas.process_full_target(Z_target=Z_blurred, dz=dz, config=config, postprocess=postprocess, verbose=False, slice_mode="envelope")

vas.evaluate_accuracy(Z_blurred, Z_final, dwell_maps, config)

# save in current directory with filename simulation_results_sine
np.savez("simulation_results_sil_from_mill.npz",
         Z_final=Z_final,
         dwell_maps=dwell_maps,
         Z_target=Z_blurred)



# TODO config.n und resolution aus der from_mill funktion müssen gleich sein, benutze config auch in der resolution methode.
n=config.n

t_total = np.sum(
    np.stack([t.reshape(n, n) for t in dwell_maps], axis=0),
    axis=0
)  # (n,n)  # TODO sollte man die dwellmaps einfach summieren, oder immer wieder neu ansetzen?

plt.figure(figsize=(6,5))
plt.imshow(t_total, cmap='viridis', origin='lower', interpolation='nearest')
plt.colorbar(label="Dwell time [µs]")
plt.title("t_total (from SILMill)")
plt.show()


# -----------------------------------------------------
# Build a Mill from this optimized dwellmap
# -----------------------------------------------------
origin = (-10, -10)  # TODO check how the origin has to look like. probably shift by (-1/2, 1/2)-fov-dimensions? automate this as nothing works if it ain't right.
optimized_mill = MatrixMill(t_total, dx=0.05, origin=origin)


# end of test of vasile
########


circ = shapes.Circle(r=7370, center=(0,0))

spiral_style = raster_styles.two_d.Spiral(pitch=20 * U_('nm'),spiral_pitch=20 * U_('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE, direction="out-in")


site.create_pattern(
    dim_shape=circ * U_('nm'),
    mill=optimized_mill,
    raster_style=spiral_style
)

#s.plot(rasterize_pitch=Q_('0.01 µm'), plot_rasterized=True)

exported = s.export(FEIStreamFile, n_rep=5, margin=0.76) 
exported.save('sil-optimized.str')


raise Exception("end of test")
