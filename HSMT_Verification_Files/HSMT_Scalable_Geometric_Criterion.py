#!/usr/bin/env python3
"""
HSMT Phase 4 – Improved Scaling Test (Adaptive n_eigen + Two-Stage Refinement)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HSMT_Improved_Scalable_Criterion:
    def __init__(self, M=300, delta_rho=0.02, window_size=6, weight_threshold=0.80):
        self.M = M
        self.delta_rho = delta_rho
        self.rho = np.linspace(-M * delta_rho, M * delta_rho, 2 * M + 1)
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

    def _find_tightest_cluster(self, eigvals_sorted, eigvecs_sorted):
        """Find the tightest near-degenerate cluster among computed eigenvalues."""
        min_diameter = np.inf
        best_start = 0
        n = len(eigvals_sorted)
        for i in range(n - self.window_size + 1):
            window = eigvals_sorted[i : i + self.window_size]
            diffs = np.abs(window[:, np.newaxis] - window[np.newaxis, :])
            diameter = np.max(diffs)
            if diameter < min_diameter:
                min_diameter = diameter
                best_start = i
        return best_start, min_diameter

    def compute_geometric_criterion(self):
        # Stage 1: Broad search with adaptive n_eigen
        n_eigen_stage1 = max(150, int(self.M * 0.6))
        eigvals1, _ = eigs(
            self.D_sparse,
            k=n_eigen_stage1,
            which='SR',
            maxiter=50000,
            tol=1e-9
        )
        sort_idx = np.argsort(np.real(eigvals1))
        eigvals_sorted = eigvals1[sort_idx]

        # Find candidate tightest cluster from Stage 1
        best_start, min_diameter = self._find_tightest_cluster(eigvals_sorted, None)

        # Stage 2: Targeted refinement around the candidate cluster
        # Use shift-invert centered on the middle of the candidate cluster
        cluster_center = np.mean(np.real(eigvals_sorted[best_start : best_start + self.window_size]))
        n_eigen_stage2 = max(80, int(self.window_size * 3))

        eigvals2, eigvecs2 = eigs(
            self.D_sparse,
            k=n_eigen_stage2,
            sigma=cluster_center,
            which='LM',
            maxiter=30000,
            tol=1e-10
        )

        # Re-sort refined eigenvalues
        sort_idx2 = np.argsort(np.real(eigvals2))
        eigvals_refined = eigvals2[sort_idx2]
        eigvecs_refined = eigvecs2[:, sort_idx2]

        # Final tightest cluster from refined set
        best_start2, min_diameter2 = self._find_tightest_cluster(eigvals_refined, eigvecs_refined)
        cluster_idx = np.arange(best_start2, best_start2 + self.window_size)

        # Compute weights from refined eigenvectors
        weights = np.sum(np.abs(eigvecs_refined[:, cluster_idx])**2, axis=1)
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
            "N_sites_total": self.N_sites,
            "num_sites_central": len(idx),
            "Delta_rho": Delta_rho,
            "H_bounded": H_bounded,
            "C": C,
            "G": G_bounded,
            "one_over_G": 1.0 / G_bounded if G_bounded > 0 else np.nan,
            "cluster_diameter": min_diameter2,
            "n_eigen_stage1": n_eigen_stage1,
            "n_eigen_stage2": n_eigen_stage2
        }


# ============================================================
# Scaling Test with Two-Stage Refinement
# ============================================================
if __name__ == "__main__":
    print("=" * 95)
    print("HSMT Phase 4 – Improved Scaling Test (Adaptive n_eigen + Two-Stage Refinement)")
    print("Parameters: window=6, weight_threshold=0.80")
    print("=" * 95)

    results = []
    test_sizes = [300, 600, 800]

    for M in test_sizes:
        model = HSMT_Improved_Scalable_Criterion(M=M, delta_rho=0.02)

        t0 = time.perf_counter()
        model.build_sparse_master_operator()
        res = model.compute_geometric_criterion()
        elapsed = time.perf_counter() - t0

        if res is not None:
            res["wall_time_sec"] = elapsed
            results.append(res)
            print(f"\nM = {M:4d}  |  Time = {elapsed:7.2f} s  |  G = {res['G']:.4f}  |  1/G = {res['one_over_G']:.4f}")
        else:
            print(f"\nM = {M:4d}  |  [No valid central region found]")

    # Summary table
    print("\n" + "=" * 95)
    print("Summary – Improved Scaling Behavior")
    print("=" * 95)
    print(f"{'M':>6} | {'Total Sites':>12} | {'Central Sites':>14} | {'Δρ':>7} | {'G':>8} | {'1/G':>8} | {'Time (s)':>10}")
    print("-" * 95)
    for r in results:
        print(f"{r['M']:6d} | {r['N_sites_total']:12d} | {r['num_sites_central']:14d} | "
              f"{r['Delta_rho']:7.2f} | {r['G']:8.4f} | {r['one_over_G']:8.4f} | {r['wall_time_sec']:10.2f}")
    print("=" * 95)