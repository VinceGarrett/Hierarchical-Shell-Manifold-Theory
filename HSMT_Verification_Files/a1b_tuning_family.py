"""A1b tuning-family identifiability calculator.

Frozen convention:
    G_j(z) = 1 / (z - omega0_j - Sigma_rec,j(z))

Goal:
Show that raw residual densities can differ across tunings while a common
latent positive measure is recovered after applying the declared inverse
tuning map T_j^{-1}.

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

def lorentzian(E, center, width):
    return width / np.pi / ((E - center)**2 + width**2)

def density_from_sigma(sigma):
    return -np.imag(sigma) / np.pi

def rel_l2(a, b):
    return np.linalg.norm(a-b) / np.linalg.norm(b)

def main():
    # Shared latent measure.
    M = np.linspace(-1.0, 1.0, 5001)
    rho_star = (2 / np.pi) * np.sqrt(np.clip(1 - M**2, 0, None))

    E = np.linspace(-1.5, 1.5, 6001)
    z = E + 1j * ETA

    # Two declared tunings.
    # T_j consists of a coupling scale g_j^2 and a named parasitic pole term.
    settings = [
        dict(name="j1", omega0=0.25, g2=0.40, pole_amp=0.035, pole_center=0.85, pole_width=0.06),
        dict(name="j2", omega0=0.48, g2=0.22, pole_amp=0.060, pole_center=-0.75, pole_width=0.08),
    ]

    recovered_latent = []
    raw_residual_densities = []

    for s in settings:
        sigma_latent = stieltjes(z, M, rho_star, s["g2"])

        # Declared parasitic pole. Written in retarded form so that
        # -Im Sigma / pi is positive.
        sigma_parasitic = s["pole_amp"] / (z - s["pole_center"] + 1j * s["pole_width"])

        # Real calibration polynomial P_1.
        p1 = 0.05 + 0.015 * E

        sigma_rec = p1 + sigma_parasitic + sigma_latent
        G = 1 / (z - s["omega0"] - sigma_rec)

        # Exact inversion in C1.
        sigma_inv = z - s["omega0"] - 1 / G
        inv_err = np.max(np.abs(sigma_inv - sigma_rec))

        # Declared residual after conventional subtraction.
        sigma_r = sigma_inv - p1 - sigma_parasitic
        rho_r = density_from_sigma(sigma_r)
        raw_residual_densities.append(rho_r)

        # Undo declared coupling transformation T_j.
        rho_latent_rec = rho_r / s["g2"]
        recovered_latent.append(rho_latent_rec)

        print(f"{s['name']} inversion max abs error: {inv_err:.3e}")
        print(f"{s['name']} raw residual weight: {trapezoid(rho_r,E):.9f}")
        print(f"{s['name']} latent-recovered weight: {trapezoid(rho_latent_rec,E):.9f}")

    raw_diff = rel_l2(raw_residual_densities[0], raw_residual_densities[1])
    latent_diff = rel_l2(recovered_latent[0], recovered_latent[1])

    # Compare against the same finite-resolution latent target.
    sigma_star_unit = stieltjes(z, M, rho_star, 1.0)
    rho_star_eta = density_from_sigma(sigma_star_unit)
    err1 = rel_l2(recovered_latent[0], rho_star_eta)
    err2 = rel_l2(recovered_latent[1], rho_star_eta)

    print(f"Raw residual density relative difference j1 vs j2: {raw_diff:.6e}")
    print(f"Recovered latent density relative difference j1 vs j2: {latent_diff:.6e}")
    print(f"j1 latent recovery relative L2 error: {err1:.6e}")
    print(f"j2 latent recovery relative L2 error: {err2:.6e}")

    # Wrong-map control: omit the coupling correction.
    wrong_map_diff = raw_diff
    print(f"Wrong-map control relative difference: {wrong_map_diff:.6e}")

    # Wrong-parasitic control: deliberately fail to subtract j2 parasitic pole.
    s2 = settings[1]
    sigma_latent2 = stieltjes(z, M, rho_star, s2["g2"])
    sigma_parasitic2 = s2["pole_amp"] / (z - s2["pole_center"] + 1j * s2["pole_width"])
    p12 = 0.05 + 0.015 * E
    sigma_rec2 = p12 + sigma_parasitic2 + sigma_latent2
    G2 = 1 / (z - s2["omega0"] - sigma_rec2)
    sigma_inv2 = z - s2["omega0"] - 1 / G2
    sigma_r2_wrong = sigma_inv2 - p12   # parasitic left undeclared
    rho_latent2_wrong = density_from_sigma(sigma_r2_wrong) / s2["g2"]
    wrong_parasitic_diff = rel_l2(recovered_latent[0], rho_latent2_wrong)
    print(f"Undeclared-parasitic latent mismatch: {wrong_parasitic_diff:.6e}")
    print(f"Undeclared-parasitic minimum latent density: {rho_latent2_wrong.min():.9f}")

    assert latent_diff < 1e-12
    assert err1 < 1e-12 and err2 < 1e-12
    assert raw_diff > 0.1
    assert wrong_parasitic_diff > 1e-2
    assert rho_latent2_wrong.min() > 0

    print("A1b tuning-family assertions: PASS")

if __name__ == "__main__":
    main()
