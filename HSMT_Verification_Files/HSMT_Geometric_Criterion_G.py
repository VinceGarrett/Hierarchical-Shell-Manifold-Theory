#!/usr/bin/env python3
"""
HSMT Phase 3 – Intrinsic Definition of Central Region
Based on real potential deviation from global mean
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HSMT_Phase3_Intrinsic_Central_Region:
    def __init__(self, M=300, delta_rho=0.02):
        self.M = M
        self.delta_rho = delta_rho
        self.rho = np.linspace(-M * delta_rho, M * delta_rho, 2 * M + 1)
        self.N_sites = len(self.rho)
        self.D_lattice = None
        self.central_indices = None

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
            V_mass = (np.pi + 0.915965594177219) * (np.pi + np.e) / 2
            D[i, i] += V_bi + V_mass

        self.D_lattice = D

    def define_intrinsic_central_region(self):
        D_real = np.real(self.D_lattice)
        diag_real = np.abs(np.diag(D_real))
        global_mean = np.mean(diag_real)
        global_std = np.std(diag_real)

        # Intrinsic definition: sites within 1 global std dev of the mean
        significant = np.abs(diag_real - global_mean) <= global_std
        self.central_indices = np.where(significant)[0]
        return self.central_indices, global_mean, global_std

    def compute_geometric_quantities(self):
        if self.central_indices is None:
            self.define_intrinsic_central_region()

        idx = self.central_indices
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
        H = mean_real / std_real if std_real > 0 else np.nan

        D_imag_block = D_imag[np.ix_(idx, idx)]
        offdiag_mask = ~np.eye(len(idx), dtype=bool)
        imag_vals = np.abs(D_imag_block[offdiag_mask])
        mean_imag = np.mean(imag_vals)
        std_imag = np.std(imag_vals)
        C = mean_imag / std_imag if std_imag > 0 else np.nan

        # Original G
        G = (Delta_rho * H * C) / mean_imag if mean_imag > 0 else np.nan

        # Refined G' (H^1.5 weighting)
        G_prime = (Delta_rho * (H ** 1.5) * C) / mean_imag if mean_imag > 0 else np.nan

        return {
            "num_sites": len(idx),
            "Delta_rho": Delta_rho,
            "H": H,
            "C": C,
            "mean_imag": mean_imag,
            "G": G,
            "G_prime": G_prime
        }


# ============================================================
# Run Intrinsic Central Region Analysis
# ============================================================
if __name__ == "__main__":
    print("=" * 82)
    print("HSMT Phase 3 – Intrinsic Definition of Central Region + Geometric Criteria")
    print("=" * 82)

    model = HSMT_Phase3_Intrinsic_Central_Region(M=300, delta_rho=0.02)
    model.build_master_operator()

    indices, gmean, gstd = model.define_intrinsic_central_region()
    results = model.compute_geometric_quantities()

    print(f"\nGlobal mean of |V_real|          : {gmean:.4f}")
    print(f"Global std dev of |V_real|       : {gstd:.4f}")
    print(f"\nIntrinsic central region sites   : {results['num_sites']}")
    print(f"Central ρ-width (Δρ)             : {results['Delta_rho']:.2f}")
    print(f"Homogeneity factor (H)           : {results['H']:.4f}")
    print(f"Coherence factor (C)             : {results['C']:.4f}")
    print(f"Mean |Im(offdiag)|               : {results['mean_imag']:.6f}")
    print(f"\nOriginal criterion G             : {results['G']:.4f}")
    print(f"Refined criterion G' (H^1.5)     : {results['G_prime']:.4f}")

    print("\n[Analysis complete]")