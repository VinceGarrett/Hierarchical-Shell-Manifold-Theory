"""A2c family-calibrated adequacy gates.

Synthetic-only validation.

For each candidate family F:
1. Generate noisy data from F under the declared noise law, eta, and window.
2. Fit all candidate families.
3. Record the best-model held-out NRMSE when F is the truth.
4. Define q95(F) from that null distribution.
5. Evaluate false-reject rates on new draws from F.
6. Evaluate false-accept rates for misspecified truths using the gate of the
   numerical winner's family.

No HSMT interpretation is used.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

SEED = 20261009
ETA = 0.02
NOISE_FRAC = 0.002
N_CAL = 30
N_TEST = 30

E = np.linspace(-3.0, 3.0, 401)
DX = E[1] - E[0]

HOLDOUT = (
    ((E > -1.15) & (E < -0.90))
    | ((E > -0.30) & (E < -0.05))
    | ((E > 0.55) & (E < 0.80))
)
TRAIN = ~HOLDOUT

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

def fit_predict(y, spec):
    fn = spec["fn"]
    # Use a fixed error scale based on peak signal rather than pointwise |y|,
    # avoiding huge weights in noisy near-zero tails.
    scale = max(np.max(np.abs(y[TRAIN])) * 0.02, 1e-12)

    def resid(theta):
        pred = blur(fn(theta))
        return (pred[TRAIN] - y[TRAIN]) / scale

    fit = least_squares(
        resid,
        spec["x0"],
        bounds=(spec["lo"], spec["hi"]),
        max_nfev=120,
        xtol=1e-9, ftol=1e-9, gtol=1e-9,
    )
    return blur(fn(fit.x))

def nrmse(pred, y):
    num = np.sqrt(np.mean((pred[HOLDOUT] - y[HOLDOUT])**2))
    den = np.sqrt(np.mean(y[HOLDOUT]**2))
    return num / den

def best_candidate(y):
    out = {}
    for name, spec in MODELS.items():
        pred = fit_predict(y, spec)
        out[name] = nrmse(pred, y)
    winner = min(out, key=out.get)
    return winner, out[winner], out

def add_noise(y, rng):
    sigma = NOISE_FRAC * np.max(y)
    return y + rng.normal(0.0, sigma, size=y.size)

# Correctly specified truths.
TRUTH_CORRECT = {
    "compact_beta": blur(compact_beta(np.array([0.72, 0.0, 1.0, 0.5]))),
    "gaussian": blur(gaussian(np.array([0.72, 0.05, 0.58]))),
    "lorentzian": blur(lorentzian(np.array([0.72, -0.05, 0.45]))),
}

# Misspecified truths.
mix = (
    0.52 * np.exp(-0.5 * ((E + 0.46) / 0.34)**2)
    + 0.38 * np.exp(-0.5 * ((E - 0.58) / 0.22)**2)
)
x2 = (E - 0.05) / 1.05
asym = np.zeros_like(E)
inside = np.abs(x2) < 1
left = inside & (x2 < 0)
right = inside & (x2 >= 0)
asym[left] = 0.78 * np.maximum(1 - x2[left]**2, 0.0)**0.35
asym[right] = 0.78 * np.maximum(1 - x2[right]**2, 0.0)**1.10

TRUTH_MISS = {
    "two_gaussian": blur(mix),
    "asym_compact": blur(asym),
}

def main():
    rng = np.random.default_rng(SEED)

    # Null calibration: q95 for each family when that family is truth and winner.
    gates = {}
    cal_errors = {}

    print("A2c family-calibrated null gates")
    print(f"seed={SEED}, eta={ETA}, noise_frac={NOISE_FRAC}, N_CAL={N_CAL}, N_TEST={N_TEST}")
    print("noise law: additive Gaussian RMS = noise_frac * peak blurred signal")

    for family, y0 in TRUTH_CORRECT.items():
        errs = []
        winner_counts = {k: 0 for k in MODELS}
        for _ in range(N_CAL):
            y = add_noise(y0, rng)
            winner, err, _ = best_candidate(y)
            winner_counts[winner] += 1
            if winner == family:
                errs.append(err)

        if len(errs) < max(5, N_CAL // 3):
            raise RuntimeError(f"too few self-wins to calibrate {family}: {len(errs)}")

        errs = np.asarray(errs)
        q95 = float(np.quantile(errs, 0.95))
        gates[family] = q95
        cal_errors[family] = errs

        print(f"\nCAL {family}")
        print(f"  self-wins: {len(errs)}/{N_CAL}")
        print(f"  winner_counts: {winner_counts}")
        print(f"  mean={errs.mean():.6f}, median={np.median(errs):.6f}, q95={q95:.6f}")

    # Fresh correctly specified test draws.
    print("\nCorrect-family validation")
    for family, y0 in TRUTH_CORRECT.items():
        accepts = 0
        correct_winner = 0
        errs = []
        for _ in range(N_TEST):
            y = add_noise(y0, rng)
            winner, err, _ = best_candidate(y)
            errs.append(err)
            correct_winner += int(winner == family)
            accepts += int(winner == family and err <= gates[family])
        print(
            f"  {family:13s}: correct_winner={correct_winner/N_TEST:.3f}, "
            f"accepted={accepts/N_TEST:.3f}, mean_best_err={np.mean(errs):.6f}"
        )

    # Misspecified test draws. Gate according to winner's family.
    print("\nMisspecified validation")
    for truth_name, y0 in TRUTH_MISS.items():
        false_accepts = 0
        wins = {k: 0 for k in MODELS}
        winner_errs = []
        for _ in range(N_TEST):
            y = add_noise(y0, rng)
            winner, err, _ = best_candidate(y)
            wins[winner] += 1
            winner_errs.append(err)
            false_accepts += int(err <= gates[winner])

        print(
            f"  {truth_name:13s}: false_accept={false_accepts/N_TEST:.3f}, "
            f"winner_counts={wins}, mean_best_err={np.mean(winner_errs):.6f}"
        )

    # Sanity only: gates finite and positive.
    assert all(np.isfinite(v) and v > 0 for v in gates.values())
    print("\nA2c family-calibrated assertions: PASS")
    print("Synthetic only. Gates are conditional on this family, noise law, eta, grid, and window.")

if __name__ == "__main__":
    main()
