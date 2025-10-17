import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr
from dataclasses import dataclass, field
from typing import Optional, Callable


# TODO whole package uses kind of annoying pixel-setup and is unitless. Should be unified with fibomat-unit-system for consistency and quality of life? On the other hand usability 
# without fibomat would be cool, so shouldn't use fibomat-units but should stick to pixels?

@dataclass
class ProcessConfig:
    """
    ProcessConfig unites a run's parameters. 
    Args:
    n: Amount of pixels in FOV
    dx, dy: Size of a single pixel in x/y direction in [m]
    sigma: Standarddeviation of Ionbeam in [m]
    h: Atoms per m^3
    f_xy : Ion Flux [ions / (m^2 s)]
    R: times of sigma after which Beam is assumed as zero
    Y0, p, q: Parameters from Yamamura-Formula. TODO find reasonable default parameters
    sigma_smooth: Amount of smoothing to be applied to avoid numerical artefacts. 
    use_numpy_grad: If True, numpy.gradient is used instead of spectral gradient.
    """
    n: int = 400
    dx: float = 0.025e-6
    dy: float = 0.025e-6
    sigma: float = 0.2e-6 
    h: float = 5e22 
    f_xy: np.array = np.ones((n, n)) * 1e19
    R: int = 3
    Y0: float = 2.5
    p: float = -0.5
    q: float = 0.0
    sigma_smooth: float = 1.0
    use_numpy_grad: bool = False

    # optional inputs (created in __post_init__ when omitted)
    f_xy: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None

    # derived fields (not passed to constructor)
    rpx: int = field(init=False)
    xs: np.ndarray = field(init=False)
    ys: np.ndarray = field(init=False)
    Xk: np.ndarray = field(init=False)
    Yk: np.ndarray = field(init=False)

    def __post_init__(self):
        # ensure f_xy matches n if not provided
        if self.f_xy is None:
            self.f_xy = np.ones((self.n, self.n)) * 1e19

        # compute kernel support in pixels
        self.rpx = int(np.ceil(self.R * self.sigma / self.dx))
        self.xs = np.arange(-self.rpx, self.rpx + 1) * self.dx
        self.ys = np.arange(-self.rpx, self.rpx + 1) * self.dy
        self.Xk, self.Yk = np.meshgrid(self.xs, self.ys, indexing="xy")

        # compute K if not provided
        if self.K is None:
            K = np.exp(-(self.Xk**2 + self.Yk**2) / (2 * self.sigma**2)) / (2 * np.pi * self.sigma**2)
            K /= K.sum()
            K *= self.dx * self.dy
            self.K = K


def compute_grad(Z, config: ProcessConfig, verbose=False):
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
    if config.sigma_smooth > 0:
        Z = gaussian_filter(Z, sigma=config.sigma_smooth)
    if config.use_numpy_grad:
        if verbose:
            print("numpy-Option was selected. For further analysis set numpy to False.")
        return np.gradient(Z, config.dx, axis=1), np.gradient(Z, config.dy, axis=0)
    n, m = Z.shape
    kx = np.fft.fftfreq(n, d=config.dx) * 2*np.pi
    ky = np.fft.fftfreq(m, d=config.dy) * 2*np.pi
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
        axs[3].scatter(np.arange(Z.shape[1]) * config.dx, Z[mid_y, :])
        axs[3].set_title("Z Section along x-axis")
        axs[3].set_xlabel("x")
        axs[3].set_ylabel("Z")

        # dzdx in x-direction
        axs[4].scatter(np.arange(dzdx.shape[1]) * config.dx, dzdx[mid_y, :], label="FFT-Gradient", marker="x")
        # NumPy-Gradient
        dzdx_np = np.gradient(Z, config.dx, axis=1)
        axs[4].scatter(np.arange(dzdx_np.shape[1]) * config.dx, dzdx_np[mid_y, :], label="NumPy-Gradient", marker="x")
        axs[4].set_title("Gradient dz/dx (section)")
        axs[4].set_xlabel("x")
        axs[4].set_ylabel("dz/dx")
        axs[4].legend()

        # dzdy in x-direction
        axs[5].scatter(np.arange(dzdy.shape[1]) * config.dy, dzdy[mid_y, :], label="FFT-Gradient", marker="x")
        # NumPy-Gradient
        dzdy_np = np.gradient(Z, config.dy, axis=0)
        axs[5].scatter(np.arange(dzdy_np.shape[1]) * config.dx, dzdy_np[mid_y, :], label="NumPy-Gradient", marker="x")
        axs[5].set_title("Gradient dz/dy (section)")
        axs[5].set_xlabel("x")
        axs[5].set_ylabel("dz/dy")
        axs[5].legend()

        # Section in y-direction
        mid_x = Z.shape[1] // 2
        axs[6].scatter(np.arange(Z.shape[0]) * config.dy, Z[:, mid_x])
        axs[6].set_title("Section along y-axis")
        axs[6].set_xlabel("y")
        axs[6].set_ylabel("Z")

        # dzdx in y-direction
        axs[7].scatter(np.arange(dzdx.shape[0]) * config.dy, dzdx[:, mid_x], label="FFT-Gradient", marker="x")
        dzdx_np = np.gradient(Z, config.dx, axis=1)
        axs[7].scatter(np.arange(dzdx_np.shape[0]) * config.dy, dzdx_np[:, mid_x], label="NumPy-Gradient", marker="x")
        axs[7].set_title("Gradient dz/dx (section)")
        axs[7].set_xlabel("y")
        axs[7].set_ylabel("dz/dx")
        axs[7].legend()

        # dzdy in y-direction
        axs[8].scatter(np.arange(dzdy.shape[0]) * config.dy, dzdy[:, mid_x], label="FFT-Gradient", marker="x")
        dzdy_np = np.gradient(Z, config.dy, axis=0)
        axs[8].scatter(np.arange(dzdy_np.shape[0]) * config.dy, dzdy_np[:, mid_x], label="NumPy-Gradient", marker="x")
        axs[8].set_title("Gradient dz/dy (section)")
        axs[8].set_xlabel("y")
        axs[8].set_ylabel("dz/dy")
        axs[8].legend()

        plt.tight_layout()
        plt.show()
    return dzdx, dzdy


def update_S_from_Z(Z, config: ProcessConfig, verbose=False):
    """
    Calculate the sputter yield matrix from the current surface.

    Args:
    Z: Matrix of current depth at each pixel
    config: Parameter for this run united in a ProcessConfig-Object
    verbose: If True, sputter yield gets plotted

    Return:
    Matrix with the sputter yield for each pixel
    """
    dzdx, dzdy = compute_grad(Z, config) # sometimes numpy = True caused a cross aligned with the axis?
    cos_theta = 1.0 / np.sqrt(1.0 + dzdx**2 + dzdy**2)
    cos_theta = np.clip(cos_theta, 1e-3, 1.0)
    sput_yield = config.Y0 * (cos_theta**config.p) * np.exp(-config.q*(1.0/cos_theta - 1.0))
    if verbose:
        plt.imshow(sput_yield, "coolwarm")
        plt.title("Sputter Yield")
        plt.colorbar()
        plt.show()
    return sput_yield

def preprocess_Z(Z, config: ProcessConfig, verbose=False):
    """
    Blurs the target surface to a realistic surface
    Args: 
    Z: Target Surface
    sigma_phys: Sigma which should be used for blurring in physical unit, gets adapted to pixel size internally. Normally, this should be at least the standard deviation of the ion beam used.
    dx: Pixelsize in x-direction TODO technically y-direction should be included as well
    verbose: If True, Z and blurred Z are plotted

    Return: Blurred Z
    """
    Z_blur = gaussian_filter(Z, config.sigma/config.dx, mode="constant")
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
        x_axis = np.arange(Z.shape[1]) * config.dx

        orig_cut = Z[center_idx, :]
        blur_cut = Z_blur[center_idx, :]

        plt.figure(figsize=(7, 5))
        plt.plot(x_axis, orig_cut, label="Original", linewidth=2)
        plt.plot(x_axis, blur_cut, "--", label="Blurred", linewidth=2)
        plt.xlabel("x-Position")
        plt.ylabel("Depth [m]")
        plt.title("Section along x-Axis")
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return Z_blur


def process_full_target(Z_target, dz, config: ProcessConfig, postprocess, verbose=True, plot_every=10):
    
    n = config.n
    Z_current = np.zeros_like(Z_target)
    dwell_maps = []
    num_slices = int(np.ceil(Z_target.max() / dz))

    if verbose:
        print(f"Starting Slice-Simulation: {num_slices} Slices à {dz*1e9:.1f} nm")

    for s in range(num_slices):
        # targeted slice depth
        D_slice = np.clip(Z_target - Z_current, 0, dz)
        if np.all(D_slice == 0):
            if verbose: print("Target profile reached.")
            break
        D_vec = D_slice.ravel()

        S_theta = update_S_from_Z(Z_current, config)

        # Matrices for current surface profile
        def C_dot(x_vec):
            X = x_vec.reshape((n, n))
            pre = S_theta * X
            conv = fftconvolve(pre, config.K, mode='same')  # TODO checken, ab wann die Näherung mit dem S_theta rausziehen eigentlich fine ist.
            return ((config.f_xy / config.h) * conv).ravel()

        def CT_dot(y_vec):
            Y = y_vec.reshape((n, n))
            temp = config.f_xy * Y
            convT = fftconvolve(temp, np.flip(np.flip(config.K,0),1), mode='same')
            return ((S_theta * convT) / config.h).ravel()

        C_linop = LinearOperator((n*n, n*n), matvec=C_dot, rmatvec=CT_dot, dtype=np.float64)


        # LSQR solve
        res = lsqr(C_linop, D_vec, atol=1e-6, btol=1e-6, iter_lim=200)  # TODO figure out tolerances
        t_unconstr = res[0]
        t_clip = np.clip(t_unconstr, 0, None)

        t_refined = postprocess(D_vec, t_clip, C_dot, CT_dot, n)#smooth_iterative_refine(D_vec, t_clip, C_dot, CT_dot, n, lam=1e-3), None, None#fista_projected(D_vec, t_clip, L_est,C_dot, CT_dot,
                         #              maxiter=maxiter, tol_dt=1e-8, verbose=False) # tol_dt=1e-8
        dwell_maps.append(t_refined)

        # Update Surface
        Z_delta = ((config.f_xy / config.h) * fftconvolve(t_refined.reshape(n,n), config.K, mode='same')) * S_theta
        Z_current += Z_delta

        if verbose and (s % plot_every == 0 or s == num_slices-1):

            residual = Z_target - Z_delta # overall error, not slice-specific

            fig, axes = plt.subplots(1, 4, figsize=(15,5))

            im3 = axes[3].imshow(D_slice, cmap="viridis")
            axes[3].set_title(f"Target slice for {s+1}/{num_slices}")
            plt.colorbar(im3, ax=axes[3], fraction=0.046, label="Depth [m]")

            im0 = axes[0].imshow(Z_current, cmap="viridis")
            axes[0].set_title(f"Surface after {s+1}/{num_slices} slices")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, label="Depth [m]")

            im1 = axes[1].imshow(S_theta, cmap="plasma")
            axes[1].set_title("Sputter Yield $S_\\theta$")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, label="atoms/ion")

            vmax = np.max(np.abs(residual))
            im2 = axes[2].imshow(residual, cmap="RdBu", vmin=-vmax, vmax=vmax)
            axes[2].set_title("Residual (C t - D)")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, label="Depth [m]")

            plt.suptitle(f"Slice {s+1}/{num_slices}")
            plt.tight_layout()
            plt.show()

            # ---------------------------
            # Section for checking usefullness of post-processing.
            # ---------------------------
            center_idx = n // 2
            x_axis = (np.arange(n) - n//2) * config.dx * 1e6  # in µm

            # reconstruct surface from t
            Z_before = ((config.f_xy / config.h) * fftconvolve(t_clip.reshape(n,n), config.K, mode='same')) * S_theta
            Z_after  = ((config.f_xy / config.h) * fftconvolve(t_refined.reshape(n,n), config.K, mode='same')) * S_theta

            target_cut = Z_target[center_idx, :] * 1e9
            before_cut = (Z_before[center_idx, :]) * 1e9
            after_cut  = (Z_after[center_idx, :]) * 1e9

            plt.figure(figsize=(7,5))
            plt.plot(x_axis, target_cut, label="Target profile", color="black", linewidth=2)
            plt.plot(x_axis, before_cut, label="before postprocess", color="red", linestyle="--", linewidth=2)
            plt.plot(x_axis, after_cut,  label="after postprocess", color="blue", linestyle="-.", linewidth=2)
            plt.xlabel("x [µm]")
            plt.ylabel("Depth [nm]")
            plt.title(f"Section along x-axis (Slice {s+1})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return Z_current, dwell_maps


def evaluate_accuracy(Z_target, Z_final, dwell_maps, config):
    n = Z_final.shape[0]
    center_idx = n // 2
    x_axis = np.arange(n) * config.dx * 1e6  # µm
    target_cut = Z_target[center_idx, :] * 1e9  # nm
    final_cut  = Z_final[center_idx, :] * 1e9   # nm
    plt.figure(figsize=(7,5))
    plt.scatter(x_axis, target_cut, label="Target Profile", linewidth=2)
    plt.scatter(x_axis, final_cut, label="Final Profile", linestyle="--", linewidth=2)
    plt.xlabel("x-Position [µm]")
    plt.ylabel("Depth [nm]")
    plt.title("Section along x-axis")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    num_slices = len(dwell_maps)
    print(f"{num_slices} dwell maps found.")

    # show first, middle and last dwell-map
    indices_to_plot = [0, num_slices//2, num_slices-1]

    fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(5*len(indices_to_plot), 4))

    for ax, idx in zip(axes, indices_to_plot):
        t_map = dwell_maps[idx].reshape((n, n))
        im = ax.imshow(t_map, cmap="inferno")
        ax.set_title(f"Dwell map Slice {idx+1}/{num_slices}")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Dwell time")

    plt.suptitle("Selected dwell maps")
    plt.tight_layout()
    plt.show()

    # Section along x-axis through dwell-maps
    cmap = plt.get_cmap("viridis")
    num = len(dwell_maps)
    colors = cmap(np.linspace(0, 1, num))

    fig, ax = plt.subplots(figsize=(8, 6))
    for k, t_vec in enumerate(dwell_maps):
        t_map = t_vec.reshape((n, n))
        cut = t_map[center_idx, :]
        ax.scatter(x_axis, cut, color=colors[k], linewidth=1.2, alpha=0.5, marker="x")

    # colorbar to indicate slice order (use fig.colorbar / pass ax to plt.colorbar)
    norm = mpl.colors.Normalize(vmin=1, vmax=num)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.arange(1, num+1))  # values shown in colorbar
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046)
    cbar.set_label("Slice index")

    ax.set_xlabel("x-Position [µm]")
    ax.set_ylabel("Dwell time")
    ax.set_title("Section of dwell maps along x-axis")
    ax.grid(True)
    fig.tight_layout()
    plt.show()



