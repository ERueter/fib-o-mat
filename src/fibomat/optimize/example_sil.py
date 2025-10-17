import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr

import vasile_with_fft_structured as vas



# Define physical parameters for this run in a ProcessConfig-object. Here, default parameters are used.
config = vas.ProcessConfig()

############################################################################
# Define target: SIL
############################################################################
n = config.n
Z_target = np.zeros((n, n))
dx = config.dx
dy = config.dy

# Parameter
radius_sil = 2.0e-6   # inner radius
radius = 3.0e-6       # outer radius
height = 5.0e-6       # maximal depth

# Coordinate grid
x = (np.arange(n) - n//2) * dx
y = (np.arange(n) - n//2) * dy
X, Y = np.meshgrid(x, y, indexing="ij")
dist = np.sqrt(X**2 + Y**2)

# Calculate target profile
Z_target = np.zeros_like(dist)

inside_sil = dist < radius_sil
Z_target[inside_sil] = height - (height / radius_sil) * np.sqrt(radius_sil**2 - dist[inside_sil]**2)

between = (dist >= radius_sil) & (dist < radius)
Z_target[between] = height * (1 - (dist[between] - radius_sil) / (radius - radius_sil))

#############################################################################################



# Test gradient computation and decide, if numpy or spectral gradient are preferrable (this can be decided in the config)
print("Testing the gradient computation.")
_, _ = vas.compute_grad(Z_target, config, verbose=True)

# Test sputter yield computation
print("Testing the sputter yield computation.")
S_theta = vas.update_S_from_Z(Z_target, config, verbose = True)

# Apply preprocessing to the target profile, e.g. blur sharp edges
print("Blurring the target profile.")
Z_blurred = vas.preprocess_Z(Z_target, config, verbose = True)

# decide if the dwell times should be postprocessed after linear approximation
def postprocess(D_vec, t_clip, C_dot, CT_dot, n):
    return t_clip

# decide how thick a slice is supposed to be
dz = 20e-8# um pro Slice 

Z_final, dwell_maps = vas.process_full_target(Z_target=Z_blurred, dz=dz, config=config, postprocess=postprocess, verbose = False)

# check how good the algorithm performed
vas.evaluate_accuracy(Z_blurred, Z_final, dwell_maps, config)

# save in current directory with filenampe simulation_results_sil
np.savez("simulation_results_sil.npz",
         Z_final=Z_final,
         dwell_maps=dwell_maps,
         Z_target=Z_blurred)
