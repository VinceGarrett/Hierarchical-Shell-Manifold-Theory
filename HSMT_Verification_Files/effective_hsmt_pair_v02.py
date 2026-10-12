"""Effective HSMT calculator v0.2.

Pair sector on top of the verified scalar engine.
No new foundational assumption. The interaction W and the probe are inputs.

Single-mode Kerr data
---------------------
Pair basis |20>, |11>, |02>, free energies 2*omega, omega+eps, 2*eps.
Rank-one shift W = lam * |v><v|, with |v> declared.
Visible probe B is declared. It is not selected by the calculus.
"""

from __future__ import annotations

import numpy as np


def free_pair(omega: float, eps: float) -> np.ndarray:
    return np.diag([2 * omega, omega + eps, 2 * eps]).astype(float)


def rank_one(lam: float, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    return lam * np.outer(v, v), v


def pair_operator(omega: float, eps: float, lam: float, v: np.ndarray) -> np.ndarray:
    H0 = free_pair(omega, eps)
    W, v = rank_one(lam, v)
    return H0 + W, H0, W, v


def pi2(z: complex, H0: np.ndarray, v: np.ndarray) -> complex:
    R0 = np.linalg.inv(z * np.eye(3) - H0)
    return complex(v @ R0 @ v)


def t2_scalar(z: complex, lam: float, H0: np.ndarray, v: np.ndarray) -> complex:
    """T_2(z) = lam / (1 - lam Pi_2(z)), as the coefficient of |v><v|."""
    return lam / (1.0 - lam * pi2(z, H0, v))


def characteristic(z: complex, H0: np.ndarray, W: np.ndarray) -> complex:
    return complex(np.linalg.det(z * np.eye(3) - H0 - W))


def free_characteristic(z: complex, H0: np.ndarray) -> complex:
    return complex(np.linalg.det(z * np.eye(3) - H0))


def numerator(z: complex, H0: np.ndarray, v: np.ndarray) -> complex:
    """N_2 such that Pi_2 = N_2 / D_2^{(0)} and D/N = 1/Pi."""
    D = free_characteristic(z, H0)
    Pi = pi2(z, H0, v)
    if abs(Pi) < 1e-14:
        return 0j
    return D * Pi


def recover_w(z: complex, lam: float, H0: np.ndarray, v: np.ndarray) -> np.ndarray:
    """W = T (I + R0 T)^{-1} where this inverse exists."""
    R0 = np.linalg.inv(z * np.eye(3) - H0)
    coeff = t2_scalar(z, lam, H0, v)
    T = coeff * np.outer(v, v)
    return T @ np.linalg.inv(np.eye(3) + R0 @ T)


def trace_lambda(omega: float, eps: float, lam: float, v: np.ndarray) -> float:
    H, H0, W, _ = pair_operator(omega, eps, lam, v)
    energies = np.linalg.eigvalsh(H)
    return float(np.sum(energies) - np.trace(H0))


def visible_correction(z: complex, lam: float, g2: float, H0: np.ndarray, v: np.ndarray, probe: np.ndarray) -> complex:
    """Delta G = <B| R |B> - <B| R0 |B>, with B = probe * g^2.

    probe is a declared vector in the pair basis. For the visible Kerr
    benchmark, probe = (2, 0, 0) makes |B|^2 = 4 g^4 on |20>.
    """
    B = g2 * np.asarray(probe, dtype=float)
    R0 = np.linalg.inv(z * np.eye(3) - H0)
    coeff = t2_scalar(z, lam, H0, v)
    T = coeff * np.outer(v, v)
    R = R0 + R0 @ T @ R0
    return complex(B @ R @ B - B @ R0 @ B)


def fisher_lambda(z: complex, lam: float, g2: float, H0: np.ndarray, v: np.ndarray, probe: np.ndarray, noise: float = 1.0) -> float:
    """One-point Fisher from d(Delta G)/d lam. Vanishes at g = 0."""
    step = 1e-5
    up = visible_correction(z, lam + step, g2, H0, v, probe)
    dn = visible_correction(z, lam - step, g2, H0, v, probe)
    deriv = (up - dn) / (2 * step)
    return float(np.abs(deriv) ** 2 / noise**2)


def run_benchmarks() -> list[tuple[str, bool, str]]:
    results = []
    omega, eps = 1.0, 1.7
    lam = 0.35
    z = 0.4 + 0.3j
    v = np.array([1.0, 0.0, 0.0])
    H, H0, W, v = pair_operator(omega, eps, lam, v)

    # T-matrix identity.
    Pi = pi2(z, H0, v)
    T = t2_scalar(z, lam, H0, v)
    results.append(("T = lam/(1-lam Pi)", abs(T - lam / (1.0 - lam * Pi)) < 1e-12, f"T={T}"))

    # Inversion recovers W.
    W_hat = recover_w(z, lam, H0, v)
    err = np.max(np.abs(W_hat - W))
    results.append(("W inversion", err < 1e-10, f"err={err:.2e}"))

    # Trace identity, any declared direction.
    for direction in (np.array([1.0, 0.0, 0.0]), np.array([0.2, 0.3, 0.9])):
        got = trace_lambda(omega, eps, lam, direction)
        results.append((f"trace lam along {direction}", abs(got - lam) < 1e-10, f"lam={got:.6f}"))

    # Resolved-pole ratio D/N = lam.
    energies = np.linalg.eigvalsh(H)
    resolved = [e for e in energies if abs(e - (omega + eps)) > 1e-8 and abs(e - 2 * eps) > 1e-8]
    ok = True
    detail = []
    for e in resolved:
        D = free_characteristic(e + 0j, H0)
        N = numerator(e + 0j, H0, v)
        ratio = D / N
        ok = ok and abs(ratio - lam) < 1e-8
        detail.append(f"{ratio:.6f}")
    results.append(("pole ratio D/N = lam", ok, ",".join(detail)))

    # Visible Kerr benchmark: probe (2,0,0), v along |20>.
    probe = np.array([2.0, 0.0, 0.0])
    g2 = 0.2
    z_vis = 3.1 + 0.05j
    D = z_vis - 2 * omega
    P = z_vis - 2 * omega - lam
    expected = 4 * lam * g2**2 / (D * P)  # g^4 = (g2)^2 if g2 means g^2; probe already carries the 2.
    # probe = (2,0,0), B = g2 * probe, |B_20|^2 = 4 g2^2.
    # Delta = |B|^2 * lam / ((z-2w)(z-2w-lam)) = 4 g2^2 lam / (D P).
    expected = 4 * g2**2 * lam / (D * P)
    got = visible_correction(z_vis, lam, g2, H0, v, probe)
    results.append(("visible Delta G", abs(got - expected) < 1e-10, f"err={abs(got-expected):.2e}"))

    # Identifiability: F ~ g^8, and F = 0 at g = 0.
    f0 = fisher_lambda(z_vis, lam, 0.0, H0, v, probe)
    f1 = fisher_lambda(z_vis, lam, 0.2, H0, v, probe)
    f2 = fisher_lambda(z_vis, lam, 0.4, H0, v, probe)
    ratio = f2 / f1 if f1 else np.inf
    results.append(("F=0 at g=0", f0 < 1e-18, f"F={f0:.2e}"))
    results.append(("F scales as g^8", abs(ratio - 16.0) < 1e-6, f"F(2g)/F(g)={ratio:.6f}"))

    # Dark direction: trace identity holds, probe sees nothing.
    v_dark = np.array([0.0, 1.0, 0.0])
    _, H0_d, _, v_dark = pair_operator(omega, eps, lam, v_dark)
    dark = visible_correction(z_vis, lam, g2, H0_d, v_dark, probe)
    held = abs(trace_lambda(omega, eps, lam, v_dark) - lam) < 1e-10
    results.append(("dark pole: trace holds, Delta G=0", held and abs(dark) < 1e-10, f"dG={abs(dark):.2e}"))

    return results


if __name__ == "__main__":
    print("Effective HSMT calculator v0.2")
    for name, ok, detail in run_benchmarks():
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}: {detail}")
