#!/usr/bin/env python3
"""
HSMT Phase 4 – Continuum Test at M=2400 (Further Refinement)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HSMT_Continuum_Test:
    def __init__(self, M, physical_half_length=6.0, window_size=6, weight_threshold=0.80):
        self.M = M
        self.physical_half_length = physical_half_length
        self.delta_rho = physical_half_length / M
        self.rho = np.linspace(-physical_half_length, physical_half_length, 2 * M + 1)
        self.N_sites = len(self.rho)
        self.window_size = window_size
        self.weight_threshold = weight_threshold
        self.D_sparse = None

    def build_sparse_master_operator(self):
        N = self.N_sites
        h = self.delta_rho

        coeff = 1j / (12 * h)
        main = np.zeros(N, dtype=complex)
        off1 = np.full(N-1,  8*coeff, dtype=complex)
        off2 = np.full(N-2, -coeff,   dtype=complex)

        D_fd = diags(
            diagonals=[main, off1, off1.conjugate(), off2, off2.conjugate()],
            offsets=[0, 1, -1, 2, -2],
            format='csr'
        )

        V_real = np.zeros(N, dtype=complex)
        for i in range(N):
            rho = self.rho[i]
            tanh_u = np.tanh(np.pi * rho)
            V_bi = (np.pi + 0.915965594177219) * 2 * np.pi / 3 * tanh_u / 2.0
            V_mass = (np.pi + 0.915965594177219) * (np.pi + np.e) / 2
            V_real[i] = V_bi + V_mass

        D_pot = diags([V_real], offsets=[0], format='csr')
        self.D_sparse = D_fd + D_pot

    def compute_geometric_criterion(self):
        n_eigen = max(350, int(self.M * 0.55))

        eigvals, eigvecs = eigs(
            self.D_sparse,
            k=n_eigen,
            which='SR',
            maxiter=120000,
            tol=1e-10
        )

        sort_idx = np.argsort(np.real(eigvals))
        eigvals_sorted = eigvals[sort_idx]
        eigvecs_sorted = eigvecs[:, sort_idx]

        min_diameter = np.inf
        best_start = 0
        for i in range(len(eigvals_sorted) - self.window_size + 1):
            window = eigvals_sorted[i : i + self.window_size]
            diffs = np.abs(window[:, np.newaxis] - window[np.newaxis, :])
            diameter = np.max(diffs)
            if diameter < min_diameter:
                min_diameter = diameter
                best_start = i

        cluster_idx = np.arange(best_start, best_start + self.window_size)

        weights = np.sum(np.abs(eigvecs_sorted[:, cluster_idx])**2, axis=1)
        total_weight = np.sum(weights)
        sorted_weights = np.sort(weights)[::-1]
        cumulative = np.cumsum(sorted_weights) / total_weight
        cutoff_rank = np.searchsorted(cumulative, self.weight_threshold) + 1

        top_indices = np.argsort(weights)[::-1][:cutoff_rank]
        idx = np.sort(top_indices)

        if len(idx) < 5:
            return None

        rho_min = self.rho[idx[0]]
        rho_max = self.rho[idx[-1]]
        Delta_rho = rho_max - rho_min

        D_block = self.D_sparse[np.ix_(idx, idx)].toarray()
        D_real = np.real(D_block)
        D_imag = np.imag(D_block)

        diag_real = np.abs(np.diag(D_real))
        mean_real = np.mean(diag_real)
        std_real = np.std(diag_real)
        H_bounded = mean_real / (mean_real + std_real)

        offdiag_mask = ~np.eye(len(idx), dtype=bool)
        imag_vals = np.abs(D_imag[offdiag_mask])
        mean_imag = np.mean(imag_vals)
        std_imag = np.std(imag_vals)
        C = mean_imag / std_imag if std_imag > 0 else np.nan

        G_bounded = (Delta_rho * H_bounded * C) / mean_imag if mean_imag > 0 else np.nan

        return {
            "M": self.M,
            "delta_rho": self.delta_rho,
            "physical_half_length": self.physical_half_length,
            "num_sites_central": len(idx),
            "Delta_rho": Delta_rho,
            "H_bounded": H_bounded,
            "C": C,
            "G": G_bounded,
            "one_over_G": 1.0 / G_bounded if G_bounded > 0 else np.nan,
            "cluster_diameter": min_diameter,
            "n_eigen": n_eigen
        }


# ============================================================
# Run M=2400 Point
# ============================================================
if __name__ == "__main__":
    print("=" * 95)
    print("HSMT Phase 4 – Continuum Test at M=2400 (Further Refinement of Extrapolation)")
    print("Fixed physical volume: physical_half_length=6.0 (L ≈ 12)")
    print("=" * 95)

    M = 2400
    model = HSMT_Continuum_Test(M=M, physical_half_length=6.0)

    t0 = time.perf_counter()
    model.build_sparse_master_operator()
    res = model.compute_geometric_criterion()
    elapsed = time.perf_counter() - t0

    if res is not None:
        print(f"\nM = {M:4d}  |  Δρ = {res['delta_rho']:.5f}  |  Time = {elapsed:8.2f} s  |  "
              f"G = {res['G']:.4f}  |  1/G = {res['one_over_G']:.4f}")
        print(f"Central region width (Δρ): {res['Delta_rho']:.2f}")
        print(f"Number of sites in central region: {res['num_sites_central']}")
    else:
        print(f"\nM = {M:4d}  |  [No valid central region found]")

    print("\n" + "=" * 95)