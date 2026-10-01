import numpy as np
import matplotlib.pyplot as plt

from fibomat.optimize.vasile_with_fft_structured import (
    ProcessConfig,
    yamamura_sputter_yield,
    simulate_milling_from_dwell_times,
)


def disturbed_yamamura_sputter_yield(
    theta,
    Y0=1.0,
    p=-1.53,
    q=-0.175,
    amplitude=0.18,
    crossover=np.deg2rad(45.0),
):
    """Perturbed Yamamura yield that is larger at low angles and smaller at high angles.

    The modulation is chosen so that it is > 1 for small incidence angles and < 1
    for large incidence angles. This produces a realistic model mismatch without
    making the yield negative.
    """
    theta = np.asarray(theta)
    base = yamamura_sputter_yield(theta, Y0=Y0, p=p, q=q)

    # smooth crossover: at theta=0 -> 1 + amplitude, at theta >> crossover -> 1 - amplitude
    decay = np.exp(-((theta / crossover) ** 2))
    modulation = 1.0 + amplitude * (2.0 * decay - 1.0)

    return np.clip(base * modulation, 0.0, None)


def plot_sputter_yield_comparison():
    theta = np.linspace(0.0, np.deg2rad(89.0), 500)
    theta_deg = np.rad2deg(theta)

    y_ref = yamamura_sputter_yield(theta, Y0=1.75, p=-1.53, q=-0.175)
    y_dist = yamamura_sputter_yield(theta, Y0=1.3, p=-4.5, q=-1.3)#disturbed_yamamura_sputter_yield(theta, Y0=1.75, p=-1.53, q=-0.175, amplitude=0.12) #

    plt.figure(figsize=(8, 5))
    plt.plot(theta_deg, y_ref, label="Yamamura", linewidth=2)
    plt.plot(theta_deg, y_dist, label="Disturbed Yamamura", linestyle="--", linewidth=2)
    plt.xlabel("Incidence angle [deg]")
    plt.ylabel("Sputter yield")
    plt.title("Reference vs. slightly disturbed sputter yield")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# path to the paper-based target data
npz_path = r"c:\Users\erue\Documents\fibomat\51-µm-sil-parameter-from-paper.npz"

# load target + dwell maps
with np.load(npz_path) as data:
    Z_target = data["Z_target"]
    dwell_maps = data["dwell_maps"]

config = ProcessConfig(
    n=Z_target.shape[0],
    dx=20e-9,
    dy=20e-9,
    sigma=170e-9,
    h=9.6e28,
    f_xy=7.2e22,
    R=3,
    Y0=1.75,
    p=-1.53,
    q=-0.175,
    use_numpy_grad=True,
)

# plot the two sputter-yield curves in the same figure
plot_sputter_yield_comparison()

# bind the config parameters explicitly to avoid argument-mismatch errors
ref_yield = lambda theta: yamamura_sputter_yield(theta, Y0=config.Y0, p=config.p, q=config.q)
dist_yield = lambda theta: disturbed_yamamura_sputter_yield(
    theta,
    Y0=config.Y0,
    p=config.p,
    q=config.q,
    amplitude=0.12,
)
dist_yield = lambda theta: yamamura_sputter_yield(theta, Y0=1.3, p=-4.5, q=-1.3)

# simulate the milling with both yields side-by-side for direct comparison
Z_ref, Z_dist = simulate_milling_from_dwell_times(
    dwell_maps=dwell_maps,
    Z_target=Z_target,
    config=config,
    sputter_yield_func=(ref_yield, dist_yield),
    verbose=True,
    plot_every=1,
)

print("final surface shape reference:", Z_ref.shape)
print("final surface shape disturbed:", Z_dist.shape)