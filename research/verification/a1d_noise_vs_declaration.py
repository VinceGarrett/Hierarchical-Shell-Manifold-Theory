"""A1d: noise versus declaration-error Monte Carlo, revised crossover scan.

Frozen C1 convention:
    G_j(z) = 1 / (z - omega0_j - Sigma_rec,j(z))

Synthetic-only goal:
Measure when declaration errors become distinguishable from noisy correct
declarations. No HSMT interpretation.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid

SEED = 20261009
ETA = 0.02
NTRIALS = 400

def stieltjes(zvals, grid, density, weight=1.0):
    out = np.empty_like(zvals, dtype=complex)
    for i, z in enumerate(zvals):
        out[i] = weight * trapezoid(density / (z - grid), grid)
    return out

def density_from_sigma(sigma):
    return -np.imag(sigma) / np.pi

def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def add_complex_multiplicative_noise(G, sigma, rng):
    n = (rng.normal(size=G.size) + 1j * rng.normal(size=G.size)) * (sigma / np.sqrt(2))
    return G * (1 + n)

def reconstruct_latent(E, z, G_noisy, setting, g2_declared,
                       pole_center_declared, pole_width_declared, pole_amp_declared):
    sigma_inv = z - setting["omega0"] - 1 / G_noisy
    p1 = 0.05 + 0.015 * E
    sigma_par_decl = pole_amp_declared / (
        z - pole_center_declared + 1j * pole_width_declared
    )
    sigma_r = sigma_inv - p1 - sigma_par_decl
    return density_from_sigma(sigma_r) / g2_declared

def summary(x):
    q = np.quantile(x, [0.05, 0.5, 0.95])
    return float(np.mean(x)), float(q[0]), float(q[1]), float(q[2])

def main():
    rng = np.random.default_rng(SEED)

    M = np.linspace(-1.0, 1.0, 3001)
    rho_star = (2 / np.pi) * np.sqrt(np.clip(1 - M**2, 0, None))
    E = np.linspace(-1.5, 1.5, 2401)
    z = E + 1j * ETA

    j1 = dict(omega0=0.25, g2=0.40, pole_amp=0.035, pole_center=0.85, pole_width=0.06)
    j2 = dict(omega0=0.48, g2=0.22, pole_amp=0.060, pole_center=-0.75, pole_width=0.08)
    p1 = 0.05 + 0.015 * E

    def make_record(s):
        sigma_latent = stieltjes(z, M, rho_star, s["g2"])
        sigma_par = s["pole_amp"] / (z - s["pole_center"] + 1j * s["pole_width"])
        sigma_rec = p1 + sigma_par + sigma_latent
        return 1 / (z - s["omega0"] - sigma_rec)

    G1, G2 = make_record(j1), make_record(j2)

    # Errors span below and above the A1c examples.
    wrong_cases = {
        "center_0.0005": ("center", 0.0005),
        "center_0.0010": ("center", 0.0010),
        "center_0.0025": ("center", 0.0025),
        "center_0.0050": ("center", 0.0050),
        "width_1pct": ("width", 0.01),
        "width_2.5pct": ("width", 0.025),
        "width_5pct": ("width", 0.05),
        "coupling_0.1pct": ("coupling", 0.001),
        "coupling_0.5pct": ("coupling", 0.005),
        "coupling_1pct": ("coupling", 0.01),
    }

    noise_levels = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    rows = []
    print("A1d revised crossover scan")
    print(f"seed={SEED}, trials={NTRIALS}, eta={ETA}")

    for ns in noise_levels:
        correct = np.empty(NTRIALS)
        wrong = {k: np.empty(NTRIALS) for k in wrong_cases}

        for t in range(NTRIALS):
            G1n = add_complex_multiplicative_noise(G1, ns, rng)
            G2n = add_complex_multiplicative_noise(G2, ns, rng)

            rho1 = reconstruct_latent(
                E,z,G1n,j1,j1["g2"],j1["pole_center"],j1["pole_width"],j1["pole_amp"])
            rho2 = reconstruct_latent(
                E,z,G2n,j2,j2["g2"],j2["pole_center"],j2["pole_width"],j2["pole_amp"])
            correct[t] = rel_l2(rho1, rho2)

            for name,(kind,val) in wrong_cases.items():
                g2, c, w = j2["g2"], j2["pole_center"], j2["pole_width"]
                if kind == "center":
                    c += val
                elif kind == "width":
                    w *= (1 + val)
                elif kind == "coupling":
                    g2 *= (1 + val)
                rho2w = reconstruct_latent(E,z,G2n,j2,g2,c,w,j2["pole_amp"])
                wrong[name][t] = rel_l2(rho1, rho2w)

        cm,c05,c50,c95 = summary(correct)
        threshold = c95
        print(f"\nnoise={ns:.0e}: correct mean={cm:.6e}, q95={c95:.6e}")

        for name in wrong_cases:
            wm,w05,w50,w95 = summary(wrong[name])
            detection = float(np.mean(wrong[name] > threshold))
            overlap = 1.0 - detection
            rows.append((ns,name,cm,c95,wm,w05,detection,overlap))
            print(f"  {name:17s} mean={wm:.6e} q05={w05:.6e} "
                  f"detect={detection:.3f} overlap={overlap:.3f}")

    # Crossover report: first noise level with detection < 95% for each error.
    print("\nFirst tested noise level with detection < 95%:")
    for name in wrong_cases:
        rr = [r for r in rows if r[1] == name]
        cross = next((r for r in rr if r[6] < 0.95), None)
        if cross is None:
            print(f"  {name:17s}: not reached")
        else:
            print(f"  {name:17s}: noise={cross[0]:.0e}, detection={cross[6]:.3f}")

    # Sanity assertions only; no assumed crossover location.
    exact_low = [r for r in rows if r[0] == 1e-5 and r[1] in {"center_0.0050","width_5pct","coupling_1pct"}]
    assert all(r[6] > 0.99 for r in exact_low)
    assert all(np.isfinite(r[2]) and np.isfinite(r[4]) for r in rows)

    print("\nA1d revised assertions: PASS")
    print("Synthetic only. Detection thresholds are specific to this model, grid, eta, window, and noise law.")

if __name__ == "__main__":
    main()
