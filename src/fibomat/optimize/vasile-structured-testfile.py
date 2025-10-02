import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr

import vasile_with_fft_structured as vas

n = 400
Z_target = np.zeros((n, n))
dx = dy = 0.025e-6

# Parameter
radius_sil = 2.0e-6   # innerer Kugelradius
radius = 3.0e-6       # äußerer Abfallradius
height = 5.0e-6       # maximale Höhe in der Mitte

# Koordinatenraster
x = (np.arange(n) - n//2) * dx
y = (np.arange(n) - n//2) * dy
X, Y = np.meshgrid(x, y, indexing="ij")
dist = np.sqrt(X**2 + Y**2)

# Profil berechnen
Z_target = np.zeros_like(dist)

inside_sil = dist < radius_sil
Z_target[inside_sil] = height - (height / radius_sil) * np.sqrt(radius_sil**2 - dist[inside_sil]**2)

between = (dist >= radius_sil) & (dist < radius)
Z_target[between] = height * (1 - (dist[between] - radius_sil) / (radius - radius_sil))

vas.compute_grad(Z_target, dx, dy, verbose=True)