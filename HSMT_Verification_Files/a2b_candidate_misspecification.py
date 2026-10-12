"""A2b synthetic candidate-set misspecification test.

Candidate set:
    compact symmetric beta-edge
    Gaussian
    Lorentzian

Truths:
    two-Gaussian mixture
    asymmetric compact beta law

Goal:
Demonstrate that the pipeline can distinguish ranking from adequacy.
A "winner" is not accepted unless held-out NRMSE is below the preregistered
adequacy threshold.

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

# Same fixed holdout pattern as A2a.
HOLDOUT = (
    ((E > -1.15) & (E < -0.90))
    | ((E > -0.30) & (E < -0.05))
    | ((E > 0.55) & (E < 0.80))
)
TRAIN = ~HOLDOUT

# Preregistered absolute adequacy gate.
ADEQUACY_NRMSE = 0.05

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
        scale = np.maximum(np.max(y[TRAIN]) * 1e-3, y[TRAIN])
        return (pred[TRAIN] - y[TRAIN]) / scale
    fit = least_squares(
        resid, spec["x0"], bounds=(spec["lo"], spec["hi"]),
        max_nfev=700, xtol=1e-12, ftol=1e-12, gtol=1e-12
    )
    return fit.x, poisson_blur(fn(fit.x))

def normalized_rmse(pred, y, mask):
    rmse = np.sqrt(np.mean((pred[mask] - y[mask])**2))
    denom = np.sqrt(np.mean(y[mask]**2))
    return rmse / denom

def two_gaussian_truth():
    return (
        0.52 * np.exp(-0.5 * ((E + 0.46) / 0.34)**2)
        + 0.38 * np.exp(-0.5 * ((E - 0.58) / 0.22)**2)
    )

def asymmetric_compact_truth():
    c, R = 0.05, 1.05
    x = (E - c) / R
    inside = np.abs(x) < 1
    rho = np.zeros_like(E)
    # Different edge powers on left/right; outside symmetric compact-beta family.
    left = inside & (x < 0)
    right = inside & (x >= 0)
    rho[left] = 0.78 * np.maximum(1 - x[left]**2, 0.0)**0.35
    rho[right] = 0.78 * np.maximum(1 - x[right]**2, 0.0)**1.10
    return rho

def run_case(name, rho_true):
    y = poisson_blur(rho_true)
    print(f"\nCASE: {name}")
    print(f"adequacy threshold (holdout NRMSE): {ADEQUACY_NRMSE:.3f}")
    print("model, train_NRMSE, holdout_NRMSE, adequate, params")
    rows = []
    for model_name, spec in MODELS.items():
        pars, pred = fit_model(y, spec)
        train_err = normalized_rmse(pred, y, TRAIN)
        hold_err = normalized_rmse(pred, y, HOLDOUT)
        adequate = hold_err <= ADEQUACY_NRMSE
        rows.append((model_name, train_err, hold_err, adequate, pars))
        print(f"{model_name:13s}, {train_err:.6e}, {hold_err:.6e}, {adequate}, {pars}")
    rows.sort(key=lambda r: r[2])
    winner = rows[0]
    verdict = winner[0] if winner[3] else "NONE_OF_THE_ABOVE"
    print(f"best_numerical_candidate: {winner[0]}")
    print(f"adequacy_verdict: {verdict}")
    return rows, verdict

def main():
    r1, v1 = run_case("two_gaussian_truth", two_gaussian_truth())
    r2, v2 = run_case("asymmetric_compact_truth", asymmetric_compact_truth())

    assert v1 == "NONE_OF_THE_ABOVE"
    assert v2 == "NONE_OF_THE_ABOVE"
    assert r1[0][2] > ADEQUACY_NRMSE
    assert r2[0][2] > ADEQUACY_NRMSE

    print("\nA2b assertions: PASS")
    print("Synthetic misspecification test only. A least-bad model is not accepted without absolute adequacy.")

if __name__ == "__main__":
    main()
