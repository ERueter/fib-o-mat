import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr

import vasile_with_fft_structured as vas

# Define physical parameters for this run in a ProcessConfig-object. Here, default parameters are used.
config = vas.ProcessConfig(use_numpy_grad=False)

############################################################################
# Define target: Sinusoidal Wave
############################################################################
n = config.n
Z_target = np.zeros((n, n))
dx = config.dx
dy = config.dy

# Parameter
wavelength = 2.0e-6    # wavelength of the sine wave
amplitude = 2.0e-6     # amplitude of the sine wave
offset = 3.0e-6       # offset from zero to ensure positive depths

# Coordinate grid
x = (np.arange(n) - n//2) * dx
y = (np.arange(n) - n//2) * dy
X, Y = np.meshgrid(x, y, indexing="ij")

# Calculate target profile: sine wave along x-direction
Z_target = offset + amplitude * np.sin(2 * np.pi * Y / wavelength)

#############################################################################################

# Test gradient computation and decide, if numpy or spectral gradient are preferrable
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
dz = 20e-8  # um pro Slice 

Z_final, dwell_maps = vas.process_full_target(Z_target=Z_blurred, dz=dz, config=config, postprocess=postprocess, verbose = True)

# check how good the algorithm performed
vas.evaluate_accuracy(Z_blurred, Z_final, dwell_maps, config)

# save in current directory with filename simulation_results_sine
np.savez("simulation_results_sine.npz",
         Z_final=Z_final,
         dwell_maps=dwell_maps,
         Z_target=Z_blurred)