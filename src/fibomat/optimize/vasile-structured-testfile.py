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
h = 5e22                   # Atome/m^3 (du hattest 5e22; falls cm^3 verwendet wurde: s.u.)
f_xy = np.ones((n, n)) * 1e19  # Ion flux [ions / (m^2 s)]
R = 3
rpx = int(np.ceil(R * sigma / dx))
xs = np.arange(-rpx, rpx + 1) * dx
ys = np.arange(-rpx, rpx + 1) * dy
Xk, Yk = np.meshgrid(xs, ys, indexing='xy')
K = np.exp(-(Xk**2 + Yk**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
K /= K.sum()   # Normierung auf 1 TODO
K *= dx*dy

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
#dx, dy = vas.compute_grad(Z_target, dx, dy, verbose=True)

# Test sputter yield computation
#S_theta = vas.update_S_from_Z(Z_target, dx, dy, numpy=False, verbose = True)

config = vas.ProcessConfig()

# Test preprocessing 
Z_blurred = vas.preprocess_Z(Z_target, config, True)

def postprocess(D_vec, t_clip, C_dot, CT_dot, n):
    return t_clip

dz = 20e-8#50e-8  # 0.050 um pro Slice 
"""
Z_final, dwell_maps = vas.process_full_target(
    Z_blurred, dz, K, dx, dy, f_xy, h, postprocess, verbose=False, plot_every=5
)
"""


Z_final, dwell_maps = vas.process_full_target(Z_target=Z_blurred, dz=dz, config=config, postprocess=postprocess)

np.savez("simulation_results_struct.npz",
         Z_final=Z_final,
         dwell_maps=dwell_maps,
         Z_target=Z_blurred)
