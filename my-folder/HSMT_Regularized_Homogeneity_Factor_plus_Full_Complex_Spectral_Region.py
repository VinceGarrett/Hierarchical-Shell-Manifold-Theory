#!/usr/bin/env python3
"""
HSMT Phase 3 – Sensitivity to Mass Term Strength (V_mass × 1.0 vs × 1.2)
Bounded Homogeneity + Full Complex Spectral Region
"""

import numpy as np
from scipy.linalg import eig
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HSMT_Phase3_Mass_Term_Sensitivity:
    def __init__(self, M=300, delta_rho=0.02, window_size=6, weight_threshold=0.80, v_mass_scale=1.0):
        self.M = M
        self.delta_rho = delta_rho
        self.rho = np.linspace(-M * delta_rho, M * delta_rho, 2 * M + 1)
        self.N_sites = len(self.rho)
        self.D_lattice = None
        self.window_size = window_size
        self.weight_threshold = weight_threshold
        self.v_mass_scale = v_mass_scale

    def build_master_operator(self):
        N = self.N_sites
        h = self.delta_rho
        D = np.zeros((N, N), dtype=complex)
        coeff = 1j / (12 * h)

        for i in range(N):
            if i >= 2: D[i, i-2] += -coeff
            if i >= 1: D[i, i-1] += 8 * coeff
            if i < N-1: D[i, i+1] += -8 * coeff
            if i < N-2: D[i, i+2] += coeff

        for i in range(N):
            rho = self.rho[i]
            tanh_u = np.tanh(np.pi * rho)
            V_bi = (np.pi + 0.915965594177219) * 2 * np.pi / 3 * tanh_u / 2.0
            V_mass = (np.pi + 0.915965594177219) * (np.pi + np.e) / 2 * self.v_mass_scale
            D[i, i] += V_bi + V_mass

        self.D_lattice = D

    def compute_geometric_criterion(self):
        eigvals, eigvecs = eig(self.D_lattice)

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

        D_real = np.real(self.D_lattice)
        D_imag = np.imag(self.D_lattice)

        diag_real = np.abs(np.diag(D_real)[idx])
        mean_real = np.mean(diag_real)
        std_real = np.std(diag_real)

        H_bounded = mean_real / (mean_real + std_real)

        D_imag_block = D_imag[np.ix_(idx, idx)]
        offdiag_mask = ~np.eye(len(idx), dtype=bool)
        imag_vals = np.abs(D_imag_block[offdiag_mask])
        mean_imag = np.mean(imag_vals)
        std_imag = np.std(imag_vals)
        C = mean_imag / std_imag if std_imag > 0 else np.nan

        G_bounded = (Delta_rho * H_bounded * C) / mean_imag if mean_imag > 0 else np.nan
        G_prime_bounded = (Delta_rho * (H_bounded ** 1.5) * C) / mean_imag if mean_imag > 0 else np.nan

        return {
            "v_mass_scale": self.v_mass_scale,
            "num_sites": len(idx),
            "Delta_rho": Delta_rho,
            "H_bounded": H_bounded,
            "C": C,
            "mean_imag": mean_imag,
            "cluster_diameter": min_diameter,
            "G_bounded": G_bounded,
            "G_prime_bounded": G_prime_bounded
        }


# ============================================================
# Run Mass Term Sensitivity
# ============================================================
if __name__ == "__main__":
    print("=" * 82)
    print("HSMT Phase 3 – Sensitivity to Mass Term Strength (V_mass × 1.0 vs × 1.2)")
    print("Bounded Homogeneity + Full Complex Spectral Region (M=300, window=6, 80% weight)")
    print("=" * 82)

    results = []
    for scale in [1.0, 1.2]:
        model = HSMT_Phase3_Mass_Term_Sensitivity(M=300, delta_rho=0.02,
                                                   window_size=6, weight_threshold=0.80,
                                                   v_mass_scale=scale)
        model.build_master_operator()
        res = model.compute_geometric_criterion()
        if res is not None:
            results.append(res)

    print(f"\n{'V_mass ×':>8} | {'Sites':>6} | {'Δρ':>6} | {'H_bounded':>10} | {'C':>7} | {'G_bounded':>10} | {'G\'_bounded':>10}")
    print("-" * 76)
    for res in results:
        print(f"{res['v_mass_scale']:8.1f} | {res['num_sites']:6d} | {res['Delta_rho']:6.2f} | "
              f"{res['H_bounded']:10.4f} | {res['C']:7.4f} | {res['G_bounded']:10.4f} | {res['G_prime_bounded']:10.4f}")

    print("\n[Analysis complete]")
