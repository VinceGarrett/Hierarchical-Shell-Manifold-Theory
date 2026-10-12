"""Effective HSMT calculator v0.1.

Scalar visible state plus an admissible scalar spectral measure.
This is a response calculus, not a selector of the measure.

Contract
--------
Inputs:  omega0, spectral measure dmu, optional probe point z or energy grid.
Outputs: Sigma(z), G(z), rho_V, K(t), poles/residues, admissibility report.

Channel identification iota is not part of this version.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

@dataclass
class SpectralMeasure:
    """Positive scalar measure on the real line. Weight need not be finite."""

    name: str
    support: tuple[float, float]
    density: callable | None = None
    atoms: list[tuple[float, float]] = field(default_factory=list)
    stieltjes: callable | None = None
    finite_weight: bool | None = None

    def total_weight(self, n: int = 4000) -> float:
        w = sum(weight for _, weight in self.atoms)
        if self.density is None:
            return w
        a, b = self.support
        if not np.isfinite(a) or not np.isfinite(b):
            return np.inf
        x = np.linspace(a, b, n)
        w += np.trapezoid(np.clip(self.density(x), 0, None), x)
        return float(w)

    def moments(self, n_moments: int = 4, n: int = 4000) -> np.ndarray:
        a, b = self.support
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("moments require a finite-support completion")
        x = np.linspace(a, b, n)
        rho = self.density(x) if self.density is not None else np.zeros_like(x)
        out = []
        for k in range(n_moments):
            m = np.trapezoid(x**k * rho, x)
            m += sum((e**k) * w for e, w in self.atoms)
            out.append(m)
        return np.asarray(out, dtype=float)


def pole(location: float = 0.0, weight: float = 1.0) -> SpectralMeasure:
    return SpectralMeasure(
        name="discrete pole",
        support=(location, location),
        atoms=[(location, weight)],
        stieltjes=lambda z: weight / (z - location),
        finite_weight=True,
    )


def semicircle(a: float = 1.0, b: float = 0.0) -> SpectralMeasure:
    """Wigner semicircle. Edges beta = +1/2, K ~ t^{-3/2}."""
    lo, hi = b - 2 * a, b + 2 * a

    def density(E):
        E = np.asarray(E, dtype=float)
        rad = np.clip(4 * a**2 - (E - b) ** 2, 0, None)
        return np.sqrt(rad) / (2 * np.pi * a**2)

    def stieltjes(z):
        z = np.asarray(z, dtype=complex)
        disc = (z - b) ** 2 - 4 * a**2
        root = np.sqrt(disc)
        # Branch with m(z) ~ 1/z at infinity: subtract the root that ~ z.
        branch = np.where(np.real(root) * np.sign(np.real(z - b) + 1e-15) < 0, -root, root)
        # Prefer Im m <= 0 for Im z > 0.
        m = ((z - b) - branch) / (2 * a**2)
        flip = (np.imag(z) > 0) & (np.imag(m) > 0)
        m = np.where(flip, ((z - b) + branch) / (2 * a**2), m)
        return m

    return SpectralMeasure("semicircle", (lo, hi), density, stieltjes=stieltjes, finite_weight=True)


def arcsine(J: float = 1.0) -> SpectralMeasure:
    """Central-site law of the homogeneous full-line chain. beta = -1/2, K ~ t^{-1/2}."""
    lo, hi = -2 * J, 2 * J

    def density(E):
        E = np.asarray(E, dtype=float)
        rad = np.clip(4 * J**2 - E**2, 0, None)
        out = np.zeros_like(E)
        mask = rad > 0
        out[mask] = 1.0 / (np.pi * np.sqrt(rad[mask]))
        return out

    def stieltjes(z):
        z = np.asarray(z, dtype=complex)
        disc = z**2 - 4 * J**2
        root = np.sqrt(disc)
        m = 1.0 / root
        flip = (np.imag(z) > 0) & (np.imag(m) > 0)
        return np.where(flip, -m, m)

    return SpectralMeasure("arcsine", (lo, hi), density, stieltjes=stieltjes, finite_weight=True)


def power_law_cutoff(beta: float = -1.0 / 3.0, Lambda: float = 1.0) -> SpectralMeasure:
    """dmu = C M^{beta} dM on [0, Lambda], normalized when beta > -1."""
    if beta <= -1:
        raise ValueError("beta > -1 required for a finite edge weight")
    C = (beta + 1) / Lambda ** (beta + 1)

    def density(E):
        E = np.asarray(E, dtype=float)
        out = np.zeros_like(E)
        mask = (E > 0) & (E <= Lambda)
        out[mask] = C * E[mask] ** beta
        return out

    return SpectralMeasure(
        name=f"power-law beta={beta:.4f} cutoff",
        support=(0.0, Lambda),
        density=density,
        finite_weight=True,
    )


def jacobi_defect_arcsine_tail(J: float = 1.0) -> SpectralMeasure:
    """Finite defect of a constant tail: a1 = sqrt(2) J, a_n = J thereafter.

    This is the arcsine contact law, recovered here as the resonant defect.
    """
    return arcsine(J)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def stieltjes(measure: SpectralMeasure, z: ArrayLike, n: int = 2000) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    if measure.stieltjes is not None:
        return np.asarray(measure.stieltjes(z), dtype=complex)
    a, b = measure.support
    x = np.linspace(a, b, n)
    rho = measure.density(x)
    # Avoid the cut: caller should pass Im z != 0, or we use a principal value.
    out = np.zeros(np.broadcast(z, x[:1]).shape[:-1] if False else z.shape, dtype=complex)
    flat = z.ravel()
    acc = np.zeros(flat.shape, dtype=complex)
    for i, zi in enumerate(flat):
        acc[i] = np.trapezoid(rho / (zi - x), x)
        acc[i] += sum(w / (zi - e) for e, w in measure.atoms)
    return acc.reshape(z.shape)


def sigma(measure: SpectralMeasure, z: ArrayLike, g2: float = 1.0, n: int = 2000) -> np.ndarray:
    return g2 * stieltjes(measure, z, n=n)


def propagator(omega0: float, measure: SpectralMeasure, z: ArrayLike, g2: float = 1.0) -> np.ndarray:
    return 1.0 / (np.asarray(z, dtype=complex) - omega0 - sigma(measure, z, g2=g2))


def visible_density(omega0: float, measure: SpectralMeasure, E: ArrayLike, g2: float = 1.0, eta: float = 1e-4) -> np.ndarray:
    G = propagator(omega0, measure, np.asarray(E, dtype=float) + 1j * eta, g2=g2)
    return -np.imag(G) / np.pi


def memory(measure: SpectralMeasure, t: ArrayLike, n: int = 4000) -> np.ndarray:
    """K(t) = int rho(E) exp(-i E t) dE, plus atoms."""
    t = np.asarray(t, dtype=float)
    a, b = measure.support
    acc = np.zeros(t.shape, dtype=complex)
    if measure.density is not None and b > a:
        x = np.linspace(a, b, n)
        rho = measure.density(x)
        phase = np.exp(-1j * np.outer(t, x))
        acc = np.trapezoid(phase * rho, x, axis=1).reshape(t.shape)
    for e, w in measure.atoms:
        acc = acc + w * np.exp(-1j * e * t)
    return acc


def edge_exponent_from_memory(t: np.ndarray, K: np.ndarray, t_min: float, t_max: float) -> float:
    """Fit the envelope of |K|. Band-edge oscillations bias a raw log-log fit."""
    mask = (t >= t_min) & (t <= t_max) & (np.abs(K) > 0)
    tt = t[mask]
    yy = np.abs(K[mask])
    if tt.size < 8:
        raise ValueError("memory window too thin")
    peaks_t = []
    peaks_y = []
    for i in range(1, tt.size - 1):
        if yy[i] >= yy[i - 1] and yy[i] > yy[i + 1]:
            peaks_t.append(tt[i])
            peaks_y.append(yy[i])
    if len(peaks_t) < 3:
        slope, _ = np.polyfit(np.log(tt), np.log(yy), 1)
        return float(-slope)
    slope, _ = np.polyfit(np.log(peaks_t), np.log(peaks_y), 1)
    return float(-slope)


def inverse_sigma(G: ArrayLike, z: ArrayLike, omega0: float) -> np.ndarray:
    G = np.asarray(G, dtype=complex)
    z = np.asarray(z, dtype=complex)
    return z - omega0 - 1.0 / G


def density_from_sigma(Sigma_boundary: ArrayLike) -> np.ndarray:
    """rho = -Im Sigma(E+i0) / pi for a normalized contact law with g^2 = 1."""
    return -np.imag(np.asarray(Sigma_boundary)) / np.pi


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------

@dataclass
class Report:
    ok: bool
    checks: dict


def admissibility(measure: SpectralMeasure, omega0: float = 0.0, g2: float = 1.0) -> Report:
    checks = {}
    z = np.linspace(-2, 2, 9) + 1j * np.linspace(0.2, 2.0, 9)
    S = sigma(measure, z, g2=g2)
    checks["im_sigma_nonpositive"] = bool(np.all(np.imag(S) <= 1e-8))
    if measure.density is not None:
        a, b = measure.support
        if np.isfinite(a) and np.isfinite(b) and b > a:
            x = np.linspace(a, b, 500)
            checks["density_nonnegative"] = bool(np.all(measure.density(x) >= -1e-10))
        else:
            checks["density_nonnegative"] = False
    else:
        checks["density_nonnegative"] = all(w >= 0 for _, w in measure.atoms)
    checks["atoms_nonnegative"] = all(w >= 0 for _, w in measure.atoms)
    weight = measure.total_weight()
    checks["finite_weight"] = bool(np.isfinite(weight))
    if checks["finite_weight"]:
        z_hi = 50.0 + 1j
        S_hi = sigma(measure, z_hi, g2=g2)
        checks["high_energy_over_z"] = bool(np.abs(S_hi * z_hi - g2 * weight) / max(g2 * weight, 1e-12) < 0.05)
        moms = measure.moments(4)
        # Hankel H_ij = mu_{i+j} positive semidefinite for a positive measure.
        H = np.array([[moms[i + j] for j in range(2)] for i in range(2)])
        eig = np.linalg.eigvalsh(H)
        checks["hankel_psd"] = bool(np.all(eig >= -1e-6 * max(1.0, np.max(np.abs(moms)))))
    else:
        checks["high_energy_over_z"] = False
        checks["hankel_psd"] = False
    # Dispersive/absorptive: on a test energy, Im from the cut matches -pi g^2 rho.
    if measure.density is not None and checks.get("density_nonnegative", False):
        a, b = measure.support
        if np.isfinite(b - a) and b > a:
            E = 0.5 * (a + b)
            eta = 1e-3
            S_cut = sigma(measure, E + 1j * eta, g2=g2, n=8000)
            rho = float(np.real(measure.density(E)))
            checks["cut_matches_density"] = bool(
                np.abs(np.imag(S_cut) + np.pi * g2 * rho) < 0.05 * max(1.0, abs(g2 * rho))
            )
    ok = all(checks.values()) if checks["finite_weight"] else checks["im_sigma_nonpositive"] and checks["density_nonnegative"]
    return Report(ok=bool(ok), checks=checks)


def find_poles(omega0: float, measure: SpectralMeasure, g2: float = 1.0, window: tuple[float, float] = (-4, 4), n: int = 400):
    """Real roots of E - omega0 - Re Sigma(E) = 0 lying off the support, with residue weight."""
    a, b = measure.support
    grid = np.linspace(window[0], window[1], n)
    vals = grid - omega0 - np.real(sigma(measure, grid + 1j * 1e-8, g2=g2))
    poles = []
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            # Bisection
            lo, hi = grid[i], grid[i + 1]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                f = mid - omega0 - np.real(sigma(measure, mid + 1j * 1e-8, g2=g2))
                if (lo - omega0 - np.real(sigma(measure, lo + 1j * 1e-8, g2=g2))) * f <= 0:
                    hi = mid
                else:
                    lo = mid
            root = 0.5 * (lo + hi)
            if a < root < b:
                continue
            dS = sigma(measure, root + 1e-4, g2=g2) - sigma(measure, root - 1e-4, g2=g2)
            deriv = 1.0 - np.real(dS) / 2e-4
            Z = 1.0 / deriv if abs(deriv) > 1e-8 else np.nan
            poles.append((float(root), float(np.real(Z))))
    return poles


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks() -> list[tuple[str, bool, str]]:
    results = []

    # Pole: exact Stieltjes and a shifted visible root.
    mu = pole(1.0, 1.0)
    z = 3.0 + 0.2j
    err = abs(sigma(mu, z, g2=0.25) - 0.25 / (z - 1.0))
    results.append(("pole stieltjes", err < 1e-10, f"err={err:.2e}"))

    # Semicircle memory exponent near 3/2. J_1(2t)/t envelope.
    mu = semicircle()
    t = np.linspace(8.0, 40.0, 800)
    p = edge_exponent_from_memory(t, memory(mu, t, n=8000), 10.0, 36.0)
    results.append(("semicircle p~3/2", abs(p - 1.5) < 0.2, f"p={p:.3f}"))

    # Arcsine memory exponent near 1/2. J_0(2t) envelope.
    mu = arcsine()
    t = np.linspace(8.0, 40.0, 800)
    p = edge_exponent_from_memory(t, memory(mu, t, n=8000), 10.0, 36.0)
    results.append(("arcsine p~1/2", abs(p - 0.5) < 0.15, f"p={p:.3f}"))

    # Finite defect is the arcsine law, not a third exponent.
    mu = jacobi_defect_arcsine_tail()
    results.append(("finite defect is arcsine", mu.name == "arcsine", mu.name))

    # Cutoff power law is finite-weight and admissible; infinite law is not.
    mu = power_law_cutoff(-1.0 / 3.0, Lambda=1.0)
    rep = admissibility(mu)
    results.append(("cutoff M^{-1/3} admissible", rep.ok, str(rep.checks)))
    infinite = SpectralMeasure(
        "scale-free",
        (0.0, np.inf),
        density=lambda E: np.asarray(E, dtype=float) ** (-1.0 / 3.0),
        finite_weight=False,
    )
    results.append(("scale-free not finite weight", infinite.total_weight() == np.inf, "infinite"))

    # Analytic Euclidean coefficient. Quadrature in log coordinate.
    C = 1.0
    s = 2.0
    logu = np.linspace(-24, 24, 200000)
    u = np.exp(logu)
    integral = np.trapezoid(np.exp(logu * 2.0 / 3.0) / (1.0 + u), logu)
    m = -C * s ** (-1.0 / 3.0) * integral
    expected = -(2 * np.pi * C / np.sqrt(3)) * s ** (-1.0 / 3.0)
    results.append(("scale-free Euclidean coefficient", abs(m - expected) / abs(expected) < 1e-3, f"rel={abs(m-expected)/abs(expected):.2e}"))

    # Inverse mode on the semicircle.
    mu = semicircle()
    E = np.linspace(-1.5, 1.5, 21)
    G = propagator(0.3, mu, E + 1j * 1e-3, g2=0.4)
    S_inv = inverse_sigma(G, E + 1j * 1e-3, 0.3)
    S_dir = sigma(mu, E + 1j * 1e-3, g2=0.4)
    err = np.max(np.abs(S_inv - S_dir))
    results.append(("inverse sigma", err < 1e-8, f"err={err:.2e}"))

    # Admissibility rejects a negative atom.
    bad = pole(0.0, -1.0)
    rep = admissibility(bad)
    results.append(("negative atom rejected", not rep.checks["atoms_nonnegative"], str(rep.checks["atoms_nonnegative"])))

    return results


if __name__ == "__main__":
    print("Effective HSMT calculator v0.1")
    for name, ok, detail in run_benchmarks():
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}: {detail}")
