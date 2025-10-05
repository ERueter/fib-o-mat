import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr

import vasile_with_fft_structured as vas

############################################################################
# Define target 
############################################################################
n = 400
Z_target = np.zeros((n, n))
dx = dy = 0.025e-6
sigma = 0.2e-6           # Beam sigma [m]

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


# Test gradient computation
dx, dy = vas.compute_grad(Z_target, dx, dy, verbose=True)

# Test sputter yield computation
S_theta = vas.update_S_from_Z(Z_target, dx, dy, numpy=False, verbose = True)

Z_blurred = vas.preprocess_Z(Z_target, sigma, dx, True)