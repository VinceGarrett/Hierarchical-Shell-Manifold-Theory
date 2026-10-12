"""A2a synthetic latent-law discrimination.

Purpose:
Validate a preregistered A2 model-comparison pipeline on two known latent laws:
(1) compact beta-edge bath
(2) smooth noncompact Gaussian bath

A2 receives a latent density from A1. It does not revisit A1 subtraction.
All candidate families are fit through the same finite-resolution forward model.
Model ranking is by held-out prediction error, not training error.

No HSMT interpretation is used.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

ETA = 0.02
N = 2401
E = np.linspace(-3.0, 3.0, N)
DX = E[1] - E[0]

# Fixed holdout windows chosen before fitting.
HOLDOUT = (
    ((E > -1.15) & (E < -0.90))
    | ((E > -0.30) & (E < -0.05))
    | ((E > 0.55) & (E < 0.80))
)
TRAIN = ~HOLDOUT

def poisson_blur(rho: np.ndarray, eta: float = ETA) -> np.ndarray:
    x = (np.arange(N) - N // 2) * DX
    kernel = eta / np.pi / (x*x + eta*eta)
    kernel /= np.sum(kernel) * DX
    return fftconvolve(rho, kernel, mode="same") * DX

def compact_beta(theta):
    A, c, R, alpha = theta
    u = 1.0 - ((E - c) / R) ** 2
    return A * np.where(u > 0, np.maximum(u, 0.0) ** alpha, 0.0)

def gaussian(theta):
    A, c, s = theta
    return A * np.exp(-0.5 * ((E - c) / s) ** 2)

def lorentzian(theta):
    A, c, g = theta
    return A * (g*g) / ((E - c)**2 + g*g)

MODELS = {
    "compact_beta": dict(
        fn=compact_beta, x0=np.array([0.65, 0.0, 1.0, 0.5]),
        lo=np.array([0.0, -0.5, 0.35, 0.05]),
        hi=np.array([2.0, 0.5, 2.2, 2.5]),
    ),
    "gaussian": dict(
        fn=gaussian, x0=np.array([0.7, 0.0, 0.7]),
        lo=np.array([0.0, -0.5, 0.10]),
        hi=np.array([2.0, 0.5, 2.0]),
    ),
    "lorentzian": dict(
        fn=lorentzian, x0=np.array([0.7, 0.0, 0.7]),
        lo=np.array([0.0, -0.5, 0.05]),
        hi=np.array([2.0, 0.5, 2.0]),
    ),
}

def fit_model(y, spec):
    fn = spec["fn"]
    def resid(theta):
        pred = poisson_blur(fn(theta))
        # fixed scaling prevents the dense near-zero tails from dominating
        scale = np.maximum(np.max(y[TRAIN]) * 1e-3, y[TRAIN])
        return (pred[TRAIN] - y[TRAIN]) / scale
    fit = least_squares(resid, spec["x0"], bounds=(spec["lo"], spec["hi"]),
                        max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    pred = poisson_blur(fn(fit.x))
    return fit.x, pred

def normalized_rmse(pred, y, mask):
    rmse = np.sqrt(np.mean((pred[mask] - y[mask])**2))
    denom = np.sqrt(np.mean(y[mask]**2))
    return rmse / denom

def run_case(name, rho_true):
    y = poisson_blur(rho_true)
    print(f"\nCASE: {name}")
    print("model, train_NRMSE, holdout_NRMSE, params")
    rows = []
    for model_name, spec in MODELS.items():
        pars, pred = fit_model(y, spec)
        train_err = normalized_rmse(pred, y, TRAIN)
        hold_err = normalized_rmse(pred, y, HOLDOUT)
        rows.append((model_name, train_err, hold_err, pars))
        print(f"{model_name:13s}, {train_err:.6e}, {hold_err:.6e}, {pars}")
    rows.sort(key=lambda r: r[2])
    print(f"winner_by_holdout: {rows[0][0]}")
    return rows

def main():
    # Truth 1: compact law with one algebraic edge pair.
    alpha_true = 0.5
    rho_compact = compact_beta(np.array([0.72, 0.0, 1.0, alpha_true]))

    # Truth 2: ordinary smooth noncompact bath.
    rho_gauss = gaussian(np.array([0.72, 0.05, 0.58]))

    r1 = run_case("compact_beta_truth", rho_compact)
    r2 = run_case("gaussian_truth", rho_gauss)

    # Recovery of compact exponent from the winning compact model.
    compact_row = next(r for r in r1 if r[0] == "compact_beta")
    alpha_fit = compact_row[3][3]
    print(f"\ncompact truth alpha={alpha_true:.6f}, recovered alpha={alpha_fit:.6f}")

    assert r1[0][0] == "compact_beta"
    assert r2[0][0] == "gaussian"
    assert abs(alpha_fit - alpha_true) < 1e-3
    assert r1[0][2] < 1e-5
    assert r2[0][2] < 1e-5

    print("A2a assertions: PASS")
    print("Synthetic validation only. Correct family selection here is not evidence for HSMT.")

if __name__ == "__main__":
    main()
