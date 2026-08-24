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


s = Sample()

config = vas.ProcessConfig(use_numpy_grad=True, Y0=1.75, p=-1.53, q=-0.175, h=9.6e28, f_xy=7.2e22) # 0.26 mm per second of millng    material_scale=0.5575e-3
def postprocess(D_vec, t_clip, C_dot, CT_dot, n):
    return t_clip

# TODO Frage an Katja: Die sil mit maxdwell-time 10 zu machen und dann einfach sehr oft zu millen müsste eigentlich in falscher shape resultieren?
#silmill = SILMill(radius_sil=3953*U_('nm'),radius=7370*U_('nm'), min_dwell_time=0.1)  # arbeitet in µs
silmill = SILMill(radius_sil=1000*U_('nm'),radius=2000*U_('nm'), min_dwell_time=0.1)  # arbeitet in µs

# TODO silmill is larger than fov in vasile rn

radius_sil=silmill._radius_sil
radius=silmill._radius
min_dwell_time=0.025
max_dwell_time = 10

def dwell_func(point: np.ndarray) -> QuantityType:
    x, y = point[0], point[1]
    dist_sq = x * x + y * y
    dist = np.sqrt(dist_sq)

    # Shift apex downward by 10% of SIL radius
    z_shift = 50 # 50 nm

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

mill = DDDMill(dwell_func, 1)

#mill = silmill

spiral_style = raster_styles.two_d.Spiral(pitch=20 * U_('nm'),spiral_pitch=20 * U_('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE, direction="out-in")

#a, repeats_for_depth = calibrate.calibrate(rasterstyle=spiral_style)

circ = shapes.Circle(r=2000, center=(0,0))


target_depth = 1*U_('µm') # µm
#repeats = repeats_for_depth(target_depth)

#print("repeats = " + str(repeats)) # 758 repeats = a mill with max-dwelltime 10 µs has to mill 758 times in total to reach depth 7.3 µm


print("berechne target")
Z_target, dx = vas.get_target_from_mill(
    mill=mill,
    resolution=config.n,            # choose desired resolution of the target
    fov=20 * U_('µm'),         # your site FOV
    unit=U_("nm"),                 # eigentlich muss hier das site-unit stehen? #mill receives µm coordinates
    verbose=True
)

# the Z_target from vas now basically just has the dwell-time at each point - so technically it is in µs, not in µm. 
# we now basically use the mill to (by abuse of notation) encode a depth by just scaling it to the target depth.
# so the mill just stored the shape for us, not the depth (the user has to know themself what depth they want)

max_Z = np.max(Z_target)
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

print("plot kommt")
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.imshow(Z_target, cmap='viridis', origin='lower', interpolation='nearest')
plt.colorbar(label=f"Target Depth [m]")
plt.title("Generated SIL Target (from SILMill)")
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
np.savez("51-µm-sil-parameter-from-paper.npz",
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



