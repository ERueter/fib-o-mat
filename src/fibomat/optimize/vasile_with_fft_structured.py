import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr


def compute_grad(Z, dx, dy, sigma_smooth=1, numpy = False, verbose=False):
    """
    Spectral discrete gradient implemantation for increased accuracy. Using fft for simplicicy which technically isn't optimal for all data.
    Maybe switch to using https://pypi.org/project/spectral-derivatives/ later. Sometimes numpy has better accuracy and can be selected as well.

    Args: 
    Z: nd Array of the data which shall be differentiated
    dx: Step size in x direction
    dy: Step size in y direction
    sigma_smooth: Optional parameter for smoothing Z before calculating the gradient
    numpy: If True, numpy.gradient is used to calculate gradient.

    Returns:
    Gradient in x and y direction

    """
    if sigma_smooth > 0:
        Z = gaussian_filter(Z, sigma=sigma_smooth)
    if numpy:
        if verbose:
            print("numpy-Option was selected. For further analysis set numpy to False.")
        return np.gradient(Z, dx, axis=1), np.gradient(Z, dy, axis=0)
    n, m = Z.shape
    kx = np.fft.fftfreq(n, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(m, d=dy) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="xy")

    Zk = np.fft.fft2(Z)
    dzdx = np.fft.ifft2(1j * KX * Zk).real
    dzdy = np.fft.ifft2(1j * KY * Zk).real

    if verbose:
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        axs = axs.flatten()

        # plot Z
        im0 = axs[0].imshow(Z, origin="lower", cmap="viridis")
        axs[0].set_title("Z (Profile)")
        fig.colorbar(im0, ax=axs[0])

        # dzdx
        im1 = axs[1].imshow(dzdx, origin="lower", cmap="coolwarm")
        axs[1].set_title("dz/dx")
        fig.colorbar(im1, ax=axs[1])

        # dzdy
        im2 = axs[2].imshow(dzdy, origin="lower", cmap="coolwarm")
        axs[2].set_title("dz/dy")
        fig.colorbar(im2, ax=axs[2])

        # Plot section in x-direction
        mid_y = Z.shape[0] // 2
        axs[3].scatter(np.arange(Z.shape[1]) * dx, Z[mid_y, :])
        axs[3].set_title("Z Section along x-axis")
        axs[3].set_xlabel("x")
        axs[3].set_ylabel("Z")

        # dzdx in x-direction
        axs[4].scatter(np.arange(dzdx.shape[1]) * dx, dzdx[mid_y, :], label="FFT-Gradient", marker="x")
        # NumPy-Gradient
        dzdx_np = np.gradient(Z, dx, axis=1)
        axs[4].scatter(np.arange(dzdx_np.shape[1]) * dx, dzdx_np[mid_y, :], label="NumPy-Gradient", marker="x")
        axs[4].set_title("Gradient dz/dx (section)")
        axs[4].set_xlabel("x")
        axs[4].set_ylabel("dz/dx")
        axs[4].legend()

        # dzdy in x-direction
        axs[5].scatter(np.arange(dzdy.shape[1]) * dy, dzdy[mid_y, :], label="FFT-Gradient", marker="x")
        # NumPy-Gradient
        dzdy_np = np.gradient(Z, dy, axis=0)
        axs[5].scatter(np.arange(dzdy_np.shape[1]) * dx, dzdy_np[mid_y, :], label="NumPy-Gradient", marker="x")
        axs[5].set_title("Gradient dz/dx (section)")
        axs[5].set_xlabel("x")
        axs[5].set_ylabel("dz/dx")
        axs[5].legend()

        # Section in y-direction
        mid_x = Z.shape[1] // 2
        axs[6].scatter(np.arange(Z.shape[0]) * dy, Z[:, mid_x])
        axs[6].set_title("Section along y-axis")
        axs[6].set_xlabel("y")
        axs[6].set_ylabel("Z")

        # dzdx in y-direction
        axs[7].scatter(np.arange(dzdx.shape[0]) * dy, dzdx[:, mid_x], label="FFT-Gradient", marker="x")
        dzdx_np = np.gradient(Z, dx, axis=1)
        axs[7].scatter(np.arange(dzdx_np.shape[0]) * dy, dzdx_np[:, mid_x], label="NumPy-Gradient", marker="x")
        axs[7].set_title("Gradient dz/dx (section)")
        axs[7].set_xlabel("y")
        axs[7].set_ylabel("dz/dx")
        axs[7].legend()

        # dzdy in y-direction
        axs[8].scatter(np.arange(dzdy.shape[0]) * dy, dzdy[:, mid_x], label="FFT-Gradient", marker="x")
        dzdy_np = np.gradient(Z, dy, axis=0)
        axs[8].scatter(np.arange(dzdy_np.shape[0]) * dy, dzdy_np[:, mid_x], label="NumPy-Gradient", marker="x")
        axs[8].set_title("Gradient dz/dy (section)")
        axs[8].set_xlabel("y")
        axs[8].set_ylabel("dz/dy")
        axs[8].legend()

        plt.tight_layout()
        plt.show()
    return dzdx, dzdy