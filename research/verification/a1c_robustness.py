"""A1c robustness calculator.

Purpose:
Quantify how errors in the declared tuning map and parasitic subtraction
degrade recovery of a common latent measure.

Frozen convention:
    G_j(z) = 1 / (z - omega0_j - Sigma_rec,j(z))

No HSMT interpretation is used.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid

ETA = 0.02

def stieltjes(zvals, grid, density, weight=1.0):
    out = np.empty_like(zvals, dtype=complex)
    for i, z in enumerate(zvals):
        out[i] = weight * trapezoid(density / (z - grid), grid)
    return out

def density_from_sigma(sigma):
    return -np.imag(sigma) / np.pi

def rel_l2(a, b):
    return np.linalg.norm(a-b) / np.linalg.norm(b)

def make_setting():
    return dict(
        omega0=0.48,
        g2=0.22,
        pole_amp=0.060,
        pole_center=-0.75,
        pole_width=0.08,
    )

def reconstruct_latent(E, z, sigma_rec, setting, g2_declared, pole_center_declared,
                       pole_width_declared, pole_amp_declared):
    p1 = 0.05 + 0.015 * E
    sigma_par_decl = pole_amp_declared / (
        z - pole_center_declared + 1j * pole_width_declared
    )
    G = 1 / (z - setting["omega0"] - sigma_rec)
    sigma_inv = z - setting["omega0"] - 1 / G
    sigma_r = sigma_inv - p1 - sigma_par_decl
    return density_from_sigma(sigma_r) / g2_declared

def main():
    # Shared latent measure.
    M = np.linspace(-1.0, 1.0, 5001)
    rho_star = (2 / np.pi) * np.sqrt(np.clip(1 - M**2, 0, None))

    E = np.linspace(-1.5, 1.5, 6001)
    z = E + 1j * ETA

    # Reference tuning j1.
    j1 = dict(
        omega0=0.25,
        g2=0.40,
        pole_amp=0.035,
        pole_center=0.85,
        pole_width=0.06,
    )

    # Test tuning j2.
    j2 = make_setting()

    # Build exact synthetic records.
    p1 = 0.05 + 0.015 * E

    sigma_latent1 = stieltjes(z, M, rho_star, j1["g2"])
    sigma_par1 = j1["pole_amp"] / (z - j1["pole_center"] + 1j * j1["pole_width"])
    sigma_rec1 = p1 + sigma_par1 + sigma_latent1

    sigma_latent2 = stieltjes(z, M, rho_star, j2["g2"])
    sigma_par2 = j2["pole_amp"] / (z - j2["pole_center"] + 1j * j2["pole_width"])
    sigma_rec2 = p1 + sigma_par2 + sigma_latent2

    # Correct latent reference from j1.
    rho1 = reconstruct_latent(
        E, z, sigma_rec1, j1,
        j1["g2"], j1["pole_center"], j1["pole_width"], j1["pole_amp"]
    )

    print("A1c coupling-map perturbation sweep")
    print("frac_error, latent_mismatch")
    coupling_results = []
    for frac in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]:
        g2_bad = j2["g2"] * (1 + frac)
        rho2_bad = reconstruct_latent(
            E, z, sigma_rec2, j2,
            g2_bad, j2["pole_center"], j2["pole_width"], j2["pole_amp"]
        )
        mismatch = rel_l2(rho1, rho2_bad)
        coupling_results.append((frac, mismatch))
        print(f"{frac:.5f}, {mismatch:.8e}")

    print("\nA1c parasitic pole-center perturbation sweep")
    print("abs_shift, latent_mismatch, min_density")
    center_results = []
    for shift in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]:
        rho2_bad = reconstruct_latent(
            E, z, sigma_rec2, j2,
            j2["g2"], j2["pole_center"] + shift,
            j2["pole_width"], j2["pole_amp"]
        )
        mismatch = rel_l2(rho1, rho2_bad)
        center_results.append((shift, mismatch, rho2_bad.min()))
        print(f"{shift:.5f}, {mismatch:.8e}, {rho2_bad.min():.8e}")

    print("\nA1c parasitic pole-width perturbation sweep")
    print("frac_error, latent_mismatch, min_density")
    width_results = []
    for frac in [0.01, 0.025, 0.05, 0.10, 0.20, 0.50]:
        width_bad = j2["pole_width"] * (1 + frac)
        rho2_bad = reconstruct_latent(
            E, z, sigma_rec2, j2,
            j2["g2"], j2["pole_center"],
            width_bad, j2["pole_amp"]
        )
        mismatch = rel_l2(rho1, rho2_bad)
        width_results.append((frac, mismatch, rho2_bad.min()))
        print(f"{frac:.5f}, {mismatch:.8e}, {rho2_bad.min():.8e}")

    # Determine rough perturbation size at which mismatch crosses 1%.
    def first_cross(results, threshold=0.01):
        for row in results:
            if row[1] >= threshold:
                return row
        return None

    print("\nFirst >=1% latent mismatch:")
    print("coupling:", first_cross(coupling_results))
    print("pole center:", first_cross(center_results))
    print("pole width:", first_cross(width_results))

    # Baseline exact recovery.
    rho2_exact = reconstruct_latent(
        E, z, sigma_rec2, j2,
        j2["g2"], j2["pole_center"], j2["pole_width"], j2["pole_amp"]
    )
    baseline = rel_l2(rho1, rho2_exact)
    print(f"\nExact-declaration baseline mismatch: {baseline:.8e}")

    assert baseline < 1e-12
    assert coupling_results[3][1] > 5e-3   # 1% coupling error should be detectable at sub-percent/percent scale
    assert center_results[-1][1] > 0.05
    assert width_results[-1][1] > 0.01

    print("A1c robustness assertions: PASS")

if __name__ == "__main__":
    main()
