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
import matplotlib.pyplot as plt

s = Sample()
spiral_style = raster_styles.two_d.Spiral(pitch=20 * U_('nm'),spiral_pitch=20 * U_('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE, direction="out-in")

data = np.load("5-µm-sil-0-5575-mm-material-rate-sputter-yield-from-newer-vasile-paper-v2.npz")
dwell_maps = data["dwell_maps"]



for i in range(0,len(dwell_maps)):
    dwell_map = dwell_maps[i]
    print(f"Layer {i}: max_d = {np.max(dwell_map):.2e} s, shape = {dwell_map.shape}")
    max_d = np.max(dwell_map)
    if max_d > 0:
        # TODO über max = 10 µs nachdenken: je mehr passes wir machen müssen, desto öfter millen wir die minimalen 100 ns rein.
        # TODO nochmal über Parameter aus Vasile sprechen: Aktuell wird an tiefster Stelle 0,17 s gemillt, was mit Vasile Parametern
        # genau hinkommt, aber deutlich mehr ist als die 1000*100 ns aus Wentaos Code
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
    
    fov_um = 20.0
    dx = fov_um / n
    origin = (-(n - 1) * dx / 2, -(n - 1) * dx / 2)  # center the matrix in the 20 µm FOV
    dwell_map_scaled *= 1e6  # convert to µs for MatrixMill TODO check this
    layer_mill = MatrixMill(dwell_map_scaled, dx=dx, origin=origin)
    
    """
    # Sample from layer_mill for visualization
    x = np.linspace(origin[0], origin[0] + (n - 1) * dx, 100)
    y = np.linspace(origin[1], origin[1] + (n - 1) * dx, 100)
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
    exported.save(f'5-µm-sil-0-5575-mm-material-rate-new-vasile-parameters-tiefer-gelegt-{i}.str')
    #s.plot(rasterize_pitch=Q_('0.01 µm'), plot_rasterized=True)
    print(f"Layer {i}: max dwell {max_d:.2f} µs, scale {scale:.4f}, n_rep {n_rep}")
    s.empty_sites()
    #exit()
