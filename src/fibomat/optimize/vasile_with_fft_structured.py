import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import lru_cache
from pathlib import Path
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, lsqr
from scipy.optimize import lsq_linear
from dataclasses import dataclass, field
from typing import Optional, Callable
from fibomat.units import QuantityType, has_time_dim, has_length_dim, Q_, U_
from fibomat.units import Q_, scale_to


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
    material_scale: Optional measured scale for the combined factor Y0 * f_xy / h.
        If provided, it overrides the analytical default and is used directly as the
        constant depth scale in the simplified milling model. # TODO make material_scale compatible with non-yamamura!!!
    sigma_smooth: Amount of smoothing to be applied to avoid numerical artefacts. 
    use_numpy_grad: If True, numpy.gradient is used instead of spectral gradient.
    """
    n: int = 1000#400
    dx: float = 20e-9#50e-9#0.025e-6 # eigentlich 20 nm
    dy: float = 20e-9#50e-9#0.025e-6
    sigma: float = 170e-9#400e-9#0.167e-6#0.2e-6  # 400 nm Halbwertsbreite
    h: float = 9.6e28 # Wert für SiC jetzt #5e28 # lets try in m^3 #5e22 # atoms/cm^3
    f_xy: int = 7.2e22 #1e19 # ions/cm^2s ?   #np.array = np.ones((n, n), dtype=np.uint8) * 1e19 # TODO save memory here
    R: int = 3
    Y0: float = 1#1.75#0.8#2.5
    p: float = -1.53#-2.5#-0.5 #das f
    q: float = -0.175#-1#0.0
    material_scale: Optional[float] = None
    sigma_smooth: float = 1.0
    use_numpy_grad: bool = True

    # optional inputs (created in __post_init__ when omitted)
    #f_xy: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None

    # derived fields (not passed to constructor)
    rpx: int = field(init=False)
    xs: np.ndarray = field(init=False)
    ys: np.ndarray = field(init=False)
    Xk: np.ndarray = field(init=False)
    Yk: np.ndarray = field(init=False)


    def __post_init__(self):
        # ensure f_xy matches n if not provided
        print("using second postinit")
        if self.f_xy is None:
            self.f_xy = 1e19

        # compute kernel support in pixels
        self.rpx = int(np.ceil(self.R * self.sigma / self.dx))
        self.xs = np.arange(-self.rpx, self.rpx + 1) * self.dx
        self.ys = np.arange(-self.rpx, self.rpx + 1) * self.dy
        self.Xk, self.Yk = np.meshgrid(self.xs, self.ys, indexing="xy")

        # compute K if not provided
        if self.K is None:
            # Gaussian kernel
            K = np.exp(-(self.Xk**2 + self.Yk**2) / (2 * self.sigma**2)) #/ (2 * np.pi * self.sigma**2) TODO wie mus dieser kernel aussehen???
            #K *= self.dx * self.dy

            # circular cutoff mask
            R_phys = self.R * self.sigma
            mask = (self.Xk**2 + self.Yk**2) <= R_phys**2
            K *= mask.astype(float)

            self.K = K

            self.material_scale = (
                self.material_scale
                if self.material_scale is not None
                else (self.f_xy / self.h)#self.Y0 * (self.f_xy / self.h)
            )

            print("sum over kernel K:", K.sum())
            t_test = np.ones((self.n, self.n))
            Z_test = fftconvolve(t_test, K) * self.material_scale
            print("mean Z_test:", Z_test.mean())



def compute_grad(Z, config: ProcessConfig, verbose=False):
    """
    Spectral discrete gradient implemantation for increased accuracy. Using fft for simplicicy which technically isn't optimal for all data.
    Maybe switch to using https://pypi.org/project/spectral-derivatives/ later. Sometimes numpy has better accuracy and can be selected as well.

    Args: 
    Z: nd Array of the data which shall be differentiated
    config: Settings for this run
    verbose: If true, comparision and results are plotted

    Returns:
    Gradient in x and y direction
    TODO not working correctly, see example_sine.py

    """
    if config.sigma_smooth > 0:
        Z = gaussian_filter(Z, sigma=config.sigma/config.dx)
    if config.use_numpy_grad:
        gradx = np.gradient(Z, config.dx, axis=1)
        grady = np.gradient(Z, config.dy, axis=0)
        if verbose:
            print("numpy-Option was selected. For further analysis set numpy to False.")
            fig, axs = plt.subplots(1, 2, figsize=(18, 12))
            axs = axs.flatten()

            # plot 
            im0 = axs[0].imshow(gradx, origin="lower", cmap="viridis")
            axs[0].set_title("dzdx")
            fig.colorbar(im0, ax=axs[0])

            # dzdy
            im1 = axs[1].imshow(grady, origin="lower", cmap="viridis")
            axs[1].set_title("dz/dy")
            fig.colorbar(im1, ax=axs[1])
            plt.tight_layout()
            plt.show()

        return gradx, grady
    n, m = Z.shape
    kx = np.fft.fftfreq(n, d=config.dx) * 2*np.pi
    ky = np.fft.fftfreq(m, d=config.dy) * 2*np.pi


    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    Zk = np.fft.fft2(Z)
    dzdx = np.fft.ifft2(1j * KY * Zk).real
    dzdy = np.fft.ifft2(1j * KX * Zk).real

    #plt.imshow(dzdx**2 + dzdy**2)
    #plt.show()

    if config.sigma_smooth > 0:
        #dzdx = gaussian_filter(dzdx, sigma=config.sigma_smooth)
        #dzdy = gaussian_filter(dzdy, sigma=config.sigma_smooth)
        pass

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

##################### Example Sputter Yield Functions #####################
def yamamura_sputter_yield(theta, Y0,p=-1.53, q=-0.175):
    """Return the Yamamura sputter yield for incidence angle ``theta``.

    ``theta`` is given in radians and may be a scalar or a NumPy array.
    """
    theta = np.asarray(theta)
    cos_theta = np.clip(np.cos(theta), 1e-3, 1.0)
    return Y0*(cos_theta**p) * np.exp(q * (1.0 / cos_theta - 1.0))



def sim_yield_eb2ev(theta_rad):
    # simulated Srimp sputteryield
    angle_deg = np.rad2deg(theta_rad)
    # Interpolate within the table; use zero above its 89-degree range.
    theta_deg = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    65, 70, 72, 74, 76, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    ], dtype=float)

    y_sim_eb2ev = np.array([
        0.8228, 0.7992, 0.8556, 0.9566, 1.0548, 1.2136, 1.4100,
        1.6456, 1.9666, 2.2688, 2.8588, 3.3954, 4.1226, 5.0692,
        6.0320, 6.2956, 6.6510, 6.8286, 6.7924, 6.5168, 6.2074,
        5.7318, 5.2194, 4.3402, 3.4274, 2.3672, 1.3926, 0.5870, 0.1792,
    ])
    return np.interp(angle_deg, theta_deg, y_sim_eb2ev,
                     left=y_sim_eb2ev[0], right=0.0)


@lru_cache(maxsize=1)
def _load_katja_sputter_yield_data():
    data_path = Path(__file__).with_name("katjas-claude-sputteryield-fit.csv")
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def katja_sputter_yield(theta_rad):
    """Interpolate Katja's fitted sputter yield for angles in radians.

    The CSV is loaded only once. Values above its 89-degree range are zero,
    matching the extrapolation policy of :func:`sim_yield_eb2ev`.
    """
    angle_deg = np.rad2deg(np.asarray(theta_rad))
    angles, yields = _load_katja_sputter_yield_data()
    return np.interp(angle_deg, angles, yields, left=yields[0], right=0.0)

########################################################################

def update_S_from_Z(
    Z,
    config: ProcessConfig,
    verbose=False,
    sputter_yield_func: Optional[Callable] = None,
):
    """
    Calculate the sputter yield matrix from the current surface.
    Divergence from the paper: Returns S_theta/Y0 because Y0 is already in material_scale.

    Args:
    Z: Matrix of current depth at each pixel
    config: Parameter for this run united in a ProcessConfig-Object
    verbose: If True, sputter yield gets plotted
    sputter_yield_func: Optional function receiving the incidence angle in
        radians and returning the sputter yield. If omitted, the Yamamura
        function is used with ``config.p`` and ``config.q``.

    Return:
    Matrix with the sputter yield for each pixel
    """
    dzdx, dzdy = compute_grad(Z, config) # sometimes numpy = True caused a cross aligned with the axis?
    cos_theta = 1.0 / np.sqrt(1.0 + dzdx**2 + dzdy**2)
    cos_theta = np.clip(cos_theta, 1e-3, 1.0)
    theta = np.arccos(cos_theta)
    if sputter_yield_func is None:
        sputter_yield_func = lambda angle: yamamura_sputter_yield(
            angle, config.Y0, config.p, config.q
        )
    sput_yield = np.asarray(sputter_yield_func(theta))
    print(dzdx.shape)
    print(cos_theta.shape)
    print(sput_yield.shape)
    if verbose:
        plt.imshow(sput_yield, "viridis")
        plt.title("Sputter Yield")
        plt.colorbar()
        plt.show()
        print("NaNs in cos_theta:", np.isnan(cos_theta).sum())
        print("min cos_theta:", np.nanmin(cos_theta))
    if config.sigma_smooth > 0:
        sput_yield = sput_yield#gaussian_filter(sput_yield, sigma=config.sigma_smooth)
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


def compute_slice_target(Z_target, Z_current, dz, slice_idx, num_slices, slice_mode="residual"):
    """
    Compute the target removal for a slice.

    Parameters:
        Z_target: final target depth matrix
        Z_current: current accumulated depth matrix
        dz: maximum depth per slice
        slice_idx: current slice index (0-based)
        num_slices: total number of slices
        slice_mode: 'residual' or 'envelope'

    The 'envelope' mode limits each point based on its own target depth
    budget across the total number of slices. This prevents shallow points
    from being removed too aggressively early.
    """
    if slice_mode == "residual":
        return np.clip(Z_target - Z_current, 0, dz)

    if slice_mode == "envelope":
        # Try successive envelopes until one produces a non-zero increment
        for k in range(slice_idx + 1, num_slices + 1):
            envelope = Z_target * k / num_slices
            diff = np.clip(envelope - Z_current, 0, dz)

            if np.any(diff > 0):
                return diff#gaussian_filter(diff, sigma=1)
        return diff

    raise ValueError(f"Unknown slice_mode: {slice_mode}")


def estimate_lipschitz(C_dot, CT_dot, N, n_iter=30):

    x = np.random.randn(N)
    x /= np.linalg.norm(x)

    for _ in range(n_iter):

        y = C_dot(x)
        x_new = CT_dot(y)

        norm = np.linalg.norm(x_new)

        x = x_new / norm
    #print(f"norm estimated: {norm}")
    return norm

def fista_projected(
    D_vec,
    x0,
    L,
    C_dot,
    CT_dot,
    maxiter=200,
    tol=1e-6,
    verbose=True
):

    x = x0.copy()
    y = x.copy()

    t = 1.0

    prev_x = x.copy()

    for k in range(maxiter):

        # Gradient
        residual = C_dot(y) - D_vec
        grad = CT_dot(residual)

        # Gradient step
        x_new = y - grad / L

        # Positivity projection
        x_new = np.maximum(x_new, 0)

        # FISTA momentum
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))

        y = x_new + ((t - 1) / t_new) * (x_new - x)

        # Convergence
        dx = np.linalg.norm(x_new - x)

        if verbose and k % 10 == 0:
            cost = 0.5 * np.linalg.norm(C_dot(x_new) - D_vec)**2
            print(f"iter {k:4d} | cost={cost:.3e} | dx={dx:.3e}")

        if dx < tol:
            if verbose:
                print(f"Converged after {k} iterations")
            break

        x = x_new
        t = t_new

    return x


def process_full_target(Z_target, dz, config: ProcessConfig, postprocess, verbose=False, plot_every=10, slice_mode="residual", record_surface_history=False, total_passes=None):
    
    n = config.n
    Z_current = np.zeros_like(Z_target, dtype=float)
    dwell_maps = []
    surface_history = []
    num_slices = int(np.ceil(Z_target.max() / dz))

    if total_passes is not None:
        passes_per_slice = total_passes // num_slices
        extra_passes = total_passes % num_slices
        if verbose:
            print(f"Using total_passes={total_passes}: {passes_per_slice} passes per slice + 1 extra for first {extra_passes} slices")

    if verbose:
        print(f"Starting Slice-Simulation: {num_slices} Slices à {dz*1e9:.1f} nm")

    for s in range(num_slices):
        repeat_count = 1
        if total_passes is not None:
            repeat_count = passes_per_slice + (1 if s < extra_passes else 0)
            if repeat_count == 0:
                if verbose:
                    print(f"No remaining passes for slice {s+1}/{num_slices}; stopping.")
                break

        # targeted slice depth
        D_slice = compute_slice_target(Z_target, Z_current, dz, s, num_slices, slice_mode=slice_mode)
        if np.all(D_slice == 0):
            if verbose: print("Target profile reached.")
            break

        if repeat_count > 1:
            D_effective = D_slice / repeat_count
        else:
            D_effective = D_slice

        D_vec = D_effective.ravel()
        
        scale = np.max(D_vec)

        D_scaled = D_vec / scale

        S_theta = update_S_from_Z(Z_current, config, sputter_yield_func=katja_sputter_yield, verbose=False)
        depth_scale = config.material_scale

        # Matrices for current surface profile
        def C_dot(x_vec):
            X = x_vec.reshape((n, n))
            conv = fftconvolve(X, config.K, mode='same')
            return (depth_scale * conv * S_theta).ravel()

        def CT_dot(y_vec):
            Y = y_vec.reshape((n, n))
            temp = depth_scale * (S_theta * Y)
            convT = fftconvolve(temp, np.flip(config.K, (0, 1)), mode='same')
            return convT.ravel()


        N = n * n

        L_est = estimate_lipschitz(
            C_dot,
            CT_dot,
            N,
            n_iter=20
        )

        print("Estimated Lipschitz:", L_est)

        if len(dwell_maps) == 0:
            t0 = np.zeros(N)
        else:
            t0 = dwell_maps[-1].ravel()

        t = fista_projected(
            D_vec=D_scaled,
            x0=t0,
            L=L_est,
            C_dot=C_dot,
            CT_dot=CT_dot,
            maxiter=25,#50,#100,#200,
            tol=1e-6,
            verbose=True
        )
        t_clip = t*scale
        
        # Apply smoothing to dwell map using beam sigma for physical consistency
        t_clip = gaussian_filter(t_clip.reshape(n,n), sigma=config.sigma/config.dx).ravel()

        t_refined = postprocess(D_vec, t_clip, C_dot, CT_dot, n)
        dwell_maps.append(t_refined)

        # Update Surface
        Z_delta = depth_scale * fftconvolve(t_refined.reshape(n,n), config.K, mode='same') * S_theta
        Z_current += repeat_count * Z_delta
        if record_surface_history:
            surface_history.append(Z_current.copy())

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
            Z_before = depth_scale * fftconvolve(t_clip.reshape(n,n), config.K, mode='same') * S_theta
            Z_after  = depth_scale * fftconvolve(t_refined.reshape(n,n), config.K, mode='same') * S_theta

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

    if record_surface_history:
        return Z_current, dwell_maps, surface_history
    return Z_current, dwell_maps


def simulate_milling_from_dwell_times(
    dwell_maps,
    Z_target,
    config: ProcessConfig,
    sputter_yield_func: Optional[Callable] = None,
    sputter_yield_func_2: Optional[Callable] = None,
    verbose=False,
    plot_every=10,
    record_surface_history=False,
):
    """
    Simulate the milling process from a sequence of dwell-time maps.

    Parameters:
        dwell_maps: Iterable of dwell-time maps. Each entry may be either a
            flattened vector of length n*n, a 2D array of shape (n, n), or a
            single 2D map for a one-step simulation.
        Z_target: Final target depth matrix used for residual/error plots.
        config: Process configuration for the forward simulation.
        sputter_yield_func: Optional angular sputter-yield function used for the
            primary forward simulation. If a 2-tuple/list is passed, it is
            interpreted as ``(yield_1, yield_2)`` and both models are simulated.
        sputter_yield_func_2: Optional second sputter-yield function for direct
            comparison against ``sputter_yield_func``.
        verbose: If True, plot intermediate simulated surfaces and the
            cross-section for each yield model.
        plot_every: Plot every nth slice when verbose is enabled.
        record_surface_history: If True, also return the history of surfaces.

    Returns:
        If only one yield function is used, returns the final surface depth matrix.
        If two yield functions are used, returns ``(Z_current_1, Z_current_2)``.
        If ``record_surface_history`` is True, also returns the history list(s).
    """
    if dwell_maps is None:
        raise ValueError("dwell_maps must not be None.")

    if isinstance(sputter_yield_func, (list, tuple)) and len(sputter_yield_func) == 2:
        sputter_yield_func_2 = sputter_yield_func[1]
        sputter_yield_func = sputter_yield_func[0]

    if sputter_yield_func is None:
        sputter_yield_func = lambda angle: yamamura_sputter_yield(
            angle, config.Y0, config.p, config.q
        )

    compare_mode = sputter_yield_func_2 is not None
    if compare_mode:
        yield_funcs = [sputter_yield_func, sputter_yield_func_2]
        yield_names = [
            getattr(f, "__name__", "yield_1") for f in yield_funcs
        ]
    else:
        yield_funcs = [sputter_yield_func]
        yield_names = [getattr(sputter_yield_func, "__name__", "yield")]

    if isinstance(dwell_maps, np.ndarray):
        if dwell_maps.ndim == 1:
            dwell_maps = [dwell_maps.reshape((config.n, config.n))]
        elif dwell_maps.ndim == 2 and dwell_maps.shape == (config.n, config.n):
            dwell_maps = [dwell_maps]
        else:
            dwell_maps = [np.asarray(dm).reshape((config.n, config.n)) for dm in dwell_maps]
    elif isinstance(dwell_maps, (list, tuple)):
        if len(dwell_maps) == 0:
            raise ValueError("dwell_maps must contain at least one dwell map.")
        dwell_maps = [np.asarray(dm).reshape((config.n, config.n)) for dm in dwell_maps]
    else:
        dwell_maps = [np.asarray(dwell_maps).reshape((config.n, config.n))]

    n = config.n
    Z_currents = [np.zeros_like(Z_target, dtype=float) for _ in yield_funcs]
    surface_history = [[] for _ in yield_funcs]

    for s, t_map in enumerate(dwell_maps):
        t_map = np.asarray(t_map, dtype=float)
        if t_map.shape != (n, n):
            raise ValueError(
                f"Each dwell map must have shape ({n}, {n}), got {t_map.shape}."
            )

        for idx, func in enumerate(yield_funcs):
            S_theta = update_S_from_Z(
                Z_currents[idx],
                config,
                sputter_yield_func=func,
                verbose=False,
            )

            depth_scale = config.material_scale
            Z_delta = depth_scale * fftconvolve(t_map, config.K, mode='same') * S_theta
            Z_currents[idx] = Z_currents[idx] + Z_delta

            if record_surface_history:
                surface_history[idx].append(Z_currents[idx].copy())

        if verbose and (s % plot_every == 0 or s == len(dwell_maps) - 1):
            center_idx = n // 2
            x_axis = (np.arange(n) - n // 2) * config.dx * 1e6  # µm
            target_cut = Z_target[center_idx, :] * 1e6  # µm

            plt.figure(figsize=(8, 5))
            plt.plot(x_axis, target_cut, label="Target profile", color="black", linewidth=2)

            for idx, Z_current in enumerate(Z_currents):
                cut = Z_current[center_idx, :] * 1e6  # µm
                plt.plot(x_axis, cut, label=f"{yield_names[idx]} profile", linewidth=2)

            plt.xlabel("x [µm]")
            plt.ylabel("Depth [µm]")
            plt.title(f"Cross-section along x-axis after step {s+1}/{len(dwell_maps)}")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

    if compare_mode:
        if record_surface_history:
            return tuple(Z_currents), tuple(surface_history)
        return tuple(Z_currents)

    if record_surface_history:
        return Z_currents[0], surface_history[0]
    return Z_currents[0]


def plot_surface_history(Z_history, Z_target, config, axis='x'):
    """
    Plot the accumulated surface profile after each slice.

    This shows the overall milled shape at the end of each slice,
    rather than the incremental slice contribution.
    """
    if len(Z_history) == 0:
        raise ValueError("Z_history must contain at least one surface snapshot.")

    n = Z_target.shape[0]
    center_idx = n // 2
    x_axis = np.arange(n) * config.dx * 1e6  # µm
    target_cut = Z_target[center_idx, :] * 1e9  # nm

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(Z_history)))

    for k, Z in enumerate(Z_history):
        cut = Z[center_idx, :] * 1e9
        alpha = 0.2 if len(Z_history) > 10 else 0.6
        linewidth = 1.0 if k not in [0, len(Z_history)//2, len(Z_history)-1] else 2.0
        label = None
        if k in [0, len(Z_history)//2, len(Z_history)-1]:
            label = f"slice {k+1}/{len(Z_history)}"
        ax.plot(x_axis, cut, color=colors[k], alpha=alpha, linewidth=linewidth, label=label)

    ax.plot(x_axis, target_cut, color="black", linewidth=2.5, linestyle="-", label="Target profile")
    ax.set_xlabel("x-Position [µm]")
    ax.set_ylabel("Depth [nm]")
    ax.set_title("Accumulated Milled Shape after Each Slice")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


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
    plt.title("Target Profile and Achieved Profile along x-Axis.")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Error heatmap
    error = Z_final - Z_target
    plt.figure(figsize=(8, 6))
    vmax = np.max(np.abs(error))
    im = plt.imshow(error, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Error [m]')
    plt.title('Error (Z_final - Z_target)')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
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

    plt.suptitle("Selected calculated dwell maps")
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
    ax.set_title("Section of calculated dwell maps along x-axis")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def get_target_from_mill(
        mill,
        resolution: int,
        fov: QuantityType,                      # Field of view (Quantity), z.B. Q_(20, "µm")
        unit=U_("µm"),                # nein, site-Größeneinheit, oder? #Interne Mill-Einheit
        verbose=True
    ):
    """
    Rastert eine Mill in ein 2D-Tiefenbild für die Vasile-Optimierung.

    Args:
        mill: Instanz von SILMill, SpecialMill oder DDDMill
        resolution: Anzahl Pixel (Bild wird resolution × resolution)
        fov: Field-of-view als Quantity (z.B. 20 µm)
        unit: Welche Einheit die Mill-Funktion erwartet (z.B. 'µm')
    Returns:
        Z_target: 2D NumPy-Array in Sekunden (float), shape (resolution,resolution)
        dx: physikalische Pixelgröße
    """

    # 1) Mill auf die gewünschte Einheit einstellen
    if hasattr(mill, "set_unit"):
        pass#mill.set_unit(unit)

    # 2) FOV und Pixelgröße bestimmen
    fov_mag = scale_to(unit, fov)  #fov.magnitude  # z.B. 20.0 für 20 µm
    print(fov_mag)
    dx = fov_mag / resolution

    # 3) Koordinatenraster (Mitte = 0,0)
    lin = (np.arange(resolution) - resolution/2 + 0.5) * dx
    X, Y = np.meshgrid(lin, lin, indexing="xy")
    print(np.max(X), np.max(Y))
    print(X)

    Z = np.zeros_like(X, dtype=float)

    # 4) Rastere Mill-Funktion (liefert Quantity)
    for j in range(resolution):
        for i in range(resolution):
            dt = mill.dwell_time(np.array([X[j,i], Y[j,i]]))  # hier kriege ich es in µs von SILMill
            Z[j,i] = scale_to(U_('µs'), dt) #scale_to(U_("ms"), dt) # Z ist dann in ms, das lassen wir jetzt mal bleiben lol gegen Einheitschaos

    # 5) Wiederholungen berücksichtigen
    if hasattr(mill, "repeats"):
        print("Wir beachten repeats der Mill!")
        Z *= mill.repeats

    if verbose:
        print(f"Generated target from mill:")
        print(f"  Resolution: {resolution}×{resolution}")
        print(f"  FOV: {fov_mag} {unit}")
        print(f"  Pixel size: {dx} {unit}")
        print(Z)

    return Z, dx

