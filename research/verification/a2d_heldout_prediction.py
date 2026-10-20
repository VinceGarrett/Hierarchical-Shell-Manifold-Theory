"""A2d: held-out record prediction.

Synthetic-only predictive validation.

Training:
- Multiple noisy records share one latent law.
- Each record has a declared multiplicative tuning map T_j: rho_j = g_j^2 * rho_*.
- Infer one latent family and one shared parameter vector using training records only.

Holdout:
- Freeze family and shared parameters.
- Apply the declared g_holdout^2 map.
- Predict the held-out record without any refit.
- Score held-out NRMSE against a null distribution generated under the same
  frozen model/noise law.

No HSMT interpretation is used.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

SEED = 20261009
ETA = 0.02
NOISE_FRAC = 0.002
N_NULL = 100

E = np.linspace(-3.0, 3.0, 501)
DX = E[1] - E[0]

x = (np.arange(E.size) - E.size // 2) * DX
kernel = ETA / np.pi / (x*x + ETA*ETA)
kernel /= np.sum(kernel) * DX

def blur(rho):
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
        fn=compact_beta,
        x0=np.array([0.65, 0.0, 1.0, 0.5]),
        lo=np.array([0.0, -0.5, 0.35, 0.05]),
        hi=np.array([2.0, 0.5, 2.2, 2.5]),
    ),
    "gaussian": dict(
        fn=gaussian,
        x0=np.array([0.7, 0.0, 0.7]),
        lo=np.array([0.0, -0.5, 0.10]),
        hi=np.array([2.0, 0.5, 2.0]),
    ),
    "lorentzian": dict(
        fn=lorentzian,
        x0=np.array([0.7, 0.0, 0.7]),
        lo=np.array([0.0, -0.5, 0.05]),
        hi=np.array([2.0, 0.5, 2.0]),
    ),
}

def add_noise(y, rng):
    sigma = NOISE_FRAC * np.max(y)
    return y + rng.normal(0.0, sigma, size=y.size)

def nrmse(pred, y):
    return np.sqrt(np.mean((pred-y)**2)) / np.sqrt(np.mean(y**2))

def fit_shared_family(records, g2s, spec):
    fn = spec["fn"]
    scales = [max(np.max(np.abs(y))*0.02, 1e-12) for y in records]

    def resid(theta):
        base = blur(fn(theta))
        parts = [(g2*base - y)/s for y, g2, s in zip(records, g2s, scales)]
        return np.concatenate(parts)

    fit = least_squares(
        resid, spec["x0"], bounds=(spec["lo"], spec["hi"]),
        max_nfev=250, xtol=1e-10, ftol=1e-10, gtol=1e-10
    )
    base = blur(fn(fit.x))
    train_errs = [nrmse(g2*base, y) for y, g2 in zip(records, g2s)]
    return fit.x, base, float(np.mean(train_errs))

def choose_family(records, g2s):
    fits = {}
    for name, spec in MODELS.items():
        pars, base, err = fit_shared_family(records, g2s, spec)
        fits[name] = (pars, base, err)
    winner = min(fits, key=lambda k: fits[k][2])
    return winner, fits[winner], fits

def main():
    rng = np.random.default_rng(SEED)

    # True latent law: compact edge, not revealed to fitting stage.
    theta_true = np.array([0.72, 0.02, 1.00, 0.55])
    rho_star = compact_beta(theta_true)
    base_true = blur(rho_star)

    # Declared tuning maps.
    g2_train = [0.30, 0.45, 0.62]
    g2_hold = 0.38

    train_records = [add_noise(g2*base_true, rng) for g2 in g2_train]
    holdout = add_noise(g2_hold*base_true, rng)

    # Fit using training records only.
    winner, (pars, base_fit, train_mean_err), fits = choose_family(train_records, g2_train)

    # Frozen prediction: no holdout refit.
    hold_pred = g2_hold * base_fit
    hold_err = nrmse(hold_pred, holdout)

    print("A2d held-out prediction")
    print(f"seed={SEED}, eta={ETA}, noise_frac={NOISE_FRAC}")
    print(f"training maps g2={g2_train}; holdout g2={g2_hold}")
    print("training family mean NRMSE:")
    for name, (_, _, err) in fits.items():
        print(f"  {name:13s}: {err:.6e}")
    print(f"selected_family: {winner}")
    print(f"fitted_parameters: {pars}")
    print(f"training_mean_NRMSE: {train_mean_err:.6e}")
    print(f"heldout_NRMSE_no_refit: {hold_err:.6e}")

    # Null distribution for heldout error under the frozen fitted law.
    null = np.empty(N_NULL)
    frozen_clean = g2_hold * base_fit
    for i in range(N_NULL):
        y = add_noise(frozen_clean, rng)
        null[i] = nrmse(frozen_clean, y)

    q95 = float(np.quantile(null, 0.95))
    p_upper = float((np.sum(null >= hold_err) + 1) / (N_NULL + 1))
    accepted = hold_err <= q95

    print(f"null_q95_heldout_NRMSE: {q95:.6e}")
    print(f"heldout_predictive_accept: {accepted}")
    print(f"upper_tail_predictive_p: {p_upper:.6f}")

    # Wrong-map control: use an incorrect holdout coupling, still no refit.
    g2_wrong = g2_hold * 1.10
    wrong_pred = g2_wrong * base_fit
    wrong_err = nrmse(wrong_pred, holdout)
    wrong_accept = wrong_err <= q95
    print(f"wrong_map_g2: {g2_wrong:.6f}")
    print(f"wrong_map_heldout_NRMSE: {wrong_err:.6e}")
    print(f"wrong_map_accept: {wrong_accept}")

    assert winner == "compact_beta"
    assert accepted
    assert wrong_err > hold_err
    assert not wrong_accept

    print("A2d assertions: PASS")
    print("Synthetic predictive validation only; no holdout information used in fitting.")

if __name__ == "__main__":
    main()
