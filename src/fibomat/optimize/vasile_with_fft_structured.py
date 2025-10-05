import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr

# TODO whole package uses kind of annoying pixel-setup and is unitless. Should be unified with fibomat-unit-system for consistency and quality of life.

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
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    Zk = np.fft.fft2(Z)
    dzdx = np.fft.ifft2(1j * KY * Zk).real
    dzdy = np.fft.ifft2(1j * KX * Zk).real

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
        axs[5].set_title("Gradient dz/dy (section)")
        axs[5].set_xlabel("x")
        axs[5].set_ylabel("dz/dy")
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


def update_S_from_Z(Z, dx, dy, Y0=2.5, p=-0.5, q=0.0, sigma_smooth=1, numpy=False, verbose=False):
    """
    Calculate the sputter yield matrix from the current surface.

    Args:
    Z: Matrix of current depth at each pixel
    dx: Pixelsize in x-direction
    dy: Pixelsize in y-direction
    Y0, p, q: Parameter from Yamamura-formula (TODO find reasonable default values)
    sigma_smooth: Amount of smoothing to be used on surface while calculating gradient
    numpy: If True, numpy.gradient is used for gradient calculation
    verbose: If True, sputter yield gets plotted

    Return:
    Matrix with the sputter yield for each pixel
    """
    dzdx, dzdy = compute_grad(Z,dx,dy, sigma_smooth, numpy) # sometimes numpy = True caused a cross aligned with the axis?
    cos_theta = 1.0 / np.sqrt(1.0 + dzdx**2 + dzdy**2)
    cos_theta = np.clip(cos_theta, 1e-3, 1.0)
    sput_yield = Y0 * (cos_theta**p) * np.exp(-q*(1.0/cos_theta - 1.0))
    if verbose:
        plt.imshow(sput_yield, "coolwarm")
        plt.title("Sputter Yield")
        plt.colorbar()
        plt.show()
    return sput_yield

def preprocess_Z(Z, sigma_phys, dx, verbose=False):
    """
    Blurs the target surface to a realistic surface
    Args: 
    Z: Target Surface
    sigma_phys: Sigma which should be used for blurring in physical unit, gets adapted to pixel size internally
    dx: Pixelsize in x-direction TODO technically y-direction should be included as well
    verbose: If True, Z and blurred Z are plotted

    Return: Blurred Z
    """
    Z_blur = gaussian_filter(Z, sigma_phys/dx, mode="constant")
    if verbose:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im0 = axes[0].imshow(Z, cmap="viridis", origin="lower")
        axes[0].set_title("Original Z_target")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, label="Depth [m]")

        im1 = axes[1].imshow(Z_blur, cmap="viridis", origin="lower")
        axes[1].set_title("Blurred Z_target")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, label="Depth [m]")

        plt.tight_layout()
        plt.show()


        center_idx = Z.shape[0] // 2
        x_axis = np.arange(Z.shape[1]) * dx

        orig_cut = Z[center_idx, :]
        blur_cut = Z_blur[center_idx, :]

        plt.figure(figsize=(7, 5))
        plt.plot(x_axis, orig_cut, label="Original", linewidth=2)
        plt.plot(x_axis, blur_cut, "--", label="Blurred", linewidth=2)
        plt.xlabel("x-Position")
        plt.ylabel("Depth [m]")
        plt.title("Section along x-Axis")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return Z_blur