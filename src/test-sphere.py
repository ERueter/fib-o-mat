from fibomat import Sample, U_, Mill, Q_, DDDMill, SILMill, MatrixMill
from fibomat import shapes, raster_styles
from fibomat import Vector
from fibomat.default_backends.fei import FEIStreamFile, SpotListBackend
import numpy as np
from fibomat.units import QuantityType
from fibomat.shapes import ParametricCurve
from fibomat.optimize import vasile_with_fft_structured as vas


s = Sample()
site = s.create_site(
    dim_position=(0, 0) * U_('µm'), dim_fov=(20, 20) * U_('µm')
)


mill = SILMill(max_dwell_time=10,radius_sil=3953*U_('nm'),radius=7370*U_('nm'),repeats=1, min_dwell_time=0.1)



# TODO: Vasile wie folgt einbauen: Die Mill muss optimiert werden!
# Dafür die vom User als Funktion übergebene Mill auf einer gewissen Genauigkeit samplen, sodass man die Tiefen-Matrix wie sonst immer für Vasile benutzt kriegt.
# Das dann wie gekannt optimieren, danach eine einfache "aus der Matrix auslesen"-Funktion als neue Mill setzen.
# TODO: units angucken. aktuell ist die Gesamtzeit am Ende bis zu 5e+9, das wirkt sehr hoch? andererseits, 5ns... im streamfile sind es 2,5 µs... aber das dann mit repeats, aber die sind hier 
# auch noch nicht drin? 
config = vas.ProcessConfig(use_numpy_grad=False)
def postprocess(D_vec, t_clip, C_dot, CT_dot, n):
    return t_clip


Z_target, dx = vas.get_target_from_mill(
    mill=mill,
    resolution=400,            # choose desired resolution of the target
    fov=20 * U_('µm'),         # your site FOV
    unit=U_("nm"),                 # eigentlich muss hier das site-unit stehen? #mill receives µm coordinates
    verbose=True
)

Z_target = config.f_xy / config.h * Z_target  # Z umrechnen von der ZEIT, die an einem Pixel verbracht werden soll, zu der TIEFE, welche die Shape da haben soll.
# andererseits ist das irgendwie doppelt? dann muss man für mill erst tiefe -> zeit rechnen und dann hier wieder zurück... aber evt. beste Lösung so.


# -----------------------------------------------------
# Plot result
# -----------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.imshow(Z_target, cmap='viridis', origin='lower', interpolation='nearest')
plt.colorbar(label="Dwell time [µs]")
plt.title("Generated SIL Target (from SILMill)")
plt.show()


dz = 20e-8  # um pro Slice 
Z_blurred = vas.preprocess_Z(Z_target, config, verbose = False)
Z_final, dwell_maps = vas.process_full_target(Z_target=Z_blurred, dz=dz, config=config, postprocess=postprocess, verbose = False)

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

s.plot(rasterize_pitch=Q_('0.01 µm'), plot_rasterized=True)

#exported = s.export(FEIStreamFile, n_rep=5, margin=0.76) 
#exported.save('sil-for-vasile.str')


raise Exception("end of test")



