#!/usr/bin/env python3
"""
HSMT Phase 5 – Sensitivity to Random Imaginary Disorder
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import time

def compute_G_with_disorder(M, disorder_strength, n_samples=5, physical_half_length=6.0, seed=42):
    np.random.seed(seed)
    delta_rho = physical_half_length / M
    rho = np.linspace(-physical_half_length, physical_half_length, 2 * M + 1)
    N = len(rho)

    results = []
    for s in range(n_samples):
        # Random imaginary disorder
        imag_disorder = np.random.normal(0.0, disorder_strength, N)

        h = delta_rho
        coeff = 1j / (12 * h)
        main = np.zeros(N, dtype=complex)
        off1 = np.full(N-1,  8*coeff, dtype=complex)
        off2 = np.full(N-2, -coeff,   dtype=complex)

        D_fd = diags(
            diagonals=[main, off1, off1.conjugate(), off2, off2.conjugate()],
            offsets=[0, 1, -1, 2, -2],
            format='csr'
        )

        V = np.zeros(N, dtype=complex)
        for i in range(N):
            r = rho[i]
            tanh_u = np.tanh(np.pi * r)
            V_bi = (np.pi + 0.915965594177219) * 2 * np.pi / 3 * tanh_u / 2.0
            V_mass = (np.pi + 0.915965594177219) * (np.pi + np.e) / 2
            V[i] = V_bi + V_mass + 1j * imag_disorder[i]

        D = D_fd + diags([V], offsets=[0], format='csr')

        n_eigen = max(350, int(M * 0.55))
        eigvals, eigvecs = eigs(D, k=n_eigen, which='SR', maxiter=150000, tol=1e-10)

        sort_idx = np.argsort(np.real(eigvals))
        eigvals_sorted = eigvals[sort_idx]
        eigvecs_sorted = eigvecs[:, sort_idx]

        min_diam = np.inf
        best_start = 0
        for i in range(len(eigvals_sorted) - 5):
            diam = np.max(np.abs(eigvals_sorted[i:i+6][:, None] - eigvals_sorted[i:i+6][None, :]))
            if diam < min_diam:
                min_diam = diam
                best_start = i

        cluster_idx = np.arange(best_start, best_start + 6)
        weights = np.sum(np.abs(eigvecs_sorted[:, cluster_idx])**2, axis=1)
        total_w = np.sum(weights)
        sorted_w = np.sort(weights)[::-1]
        cum = np.cumsum(sorted_w) / total_w
        cutoff = np.searchsorted(cum, 0.80) + 1
        top_idx = np.sort(np.argsort(weights)[::-1][:cutoff])

        if len(top_idx) < 5:
            results.append((np.nan, np.nan))
            continue

        rho_min = rho[top_idx[0]]
        rho_max = rho[top_idx[-1]]
        Delta_rho = rho_max - rho_min

        D_block = D[np.ix_(top_idx, top_idx)].toarray()
        D_real = np.real(D_block)
        D_imag = np.imag(D_block)

        diag_r = np.abs(np.diag(D_real))
        H_b = np.mean(diag_r) / (np.mean(diag_r) + np.std(diag_r))

        off_mask = ~np.eye(len(top_idx), dtype=bool)
        mean_im = np.mean(np.abs(D_imag[off_mask]))
        std_im = np.std(np.abs(D_imag[off_mask]))
        C = mean_im / std_im if std_im > 0 else np.nan

        G = (Delta_rho * H_b * C) / mean_im if mean_im > 0 else np.nan
        results.append((G, Delta_rho))

    Gs = [r[0] for r in results if not np.isnan(r[0])]
    return {
        "mean_G": np.mean(Gs),
        "std_G": np.std(Gs),
        "mean_Delta_rho": np.mean([r[1] for r in results if not np.isnan(r[1])]),
        "n_valid": len(Gs)
    }


if __name__ == "__main__":
    print("HSMT Phase 5 – Sensitivity to Random Imaginary Disorder (M=300)")
    t0 = time.perf_counter()

    for strength in [0.0, 0.2, 0.4, 0.6, 0.8]:
        res = compute_G_with_disorder(M=300, disorder_strength=strength, n_samples=5)
        print(f"Disorder strength = {strength:.1f} | "
              f"mean G = {res['mean_G']:.4f} ± {res['std_G']:.4f} | "
              f"mean Δρ = {res['mean_Delta_rho']:.2f} | valid = {res['n_valid']}/5")

    print(f"\nTotal time: {time.perf_counter() - t0:.1f} s")