#!/usr/bin/env python3
"""
HSMT Phase 3 – Collective Character of Imaginary Couplings
Central vs Outer Region Comparison
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HSMT_Phase3_Collective_Character:
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

    def identify_central_region(self, threshold=0.01):
        w = np.exp(-0.5 * (self.rho / 0.5)**2)
        significant = w > (threshold * np.max(w))
        self.central_indices = np.where(significant)[0]
        return self.central_indices

    def analyze_imaginary_character(self):
        if self.central_indices is None:
            self.identify_central_region()

        D_imag = np.imag(self.D_lattice)

        idx_c = self.central_indices
        all_idx = np.arange(self.N_sites)
        idx_o = np.setdiff1d(all_idx, idx_c)

        # Central region
        D_imag_c = D_imag[np.ix_(idx_c, idx_c)]
        offdiag_mask_c = ~np.eye(len(idx_c), dtype=bool)
        imag_vals_c = np.abs(D_imag_c[offdiag_mask_c])
        mean_c = np.mean(imag_vals_c)
        std_c = np.std(imag_vals_c)
        cv_c = std_c / mean_c if mean_c > 0 else np.nan

        # Outer region
        D_imag_o = D_imag[np.ix_(idx_o, idx_o)]
        offdiag_mask_o = ~np.eye(len(idx_o), dtype=bool)
        imag_vals_o = np.abs(D_imag_o[offdiag_mask_o])
        mean_o = np.mean(imag_vals_o)
        std_o = np.std(imag_vals_o)
        cv_o = std_o / mean_o if mean_o > 0 else np.nan

        return {
            "central": {
                "num_sites": len(idx_c),
                "mean_|Im|": mean_c,
                "std_|Im|": std_c,
                "coeff_of_variation": cv_c
            },
            "outer": {
                "num_sites": len(idx_o),
                "mean_|Im|": mean_o,
                "std_|Im|": std_o,
                "coeff_of_variation": cv_o
            },
            "cv_ratio_central_over_outer": cv_c / cv_o if cv_o > 0 else np.nan
        }


# ============================================================
# Run Collective Character Analysis
# ============================================================
if __name__ == "__main__":
    print("=" * 82)
    print("HSMT Phase 3 – Collective Character of Imaginary Couplings")
    print("Central vs Outer Region (Coefficient of Variation of |Im(offdiag)|)")
    print("=" * 82)

    model = HSMT_Phase3_Collective_Character(M=300, delta_rho=0.02)
    model.build_master_operator()
    model.identify_central_region(threshold=0.01)

    results = model.analyze_imaginary_character()

    print("\n--- Central Region ---")
    print(f"Sites                  : {results['central']['num_sites']}")
    print(f"Mean |Im(offdiag)|     : {results['central']['mean_|Im|']:.6f}")
    print(f"Std  |Im(offdiag)|     : {results['central']['std_|Im|']:.6f}")
    print(f"Coefficient of Variation: {results['central']['coeff_of_variation']:.4f}")

    print("\n--- Outer Region ---")
    print(f"Sites                  : {results['outer']['num_sites']}")
    print(f"Mean |Im(offdiag)|     : {results['outer']['mean_|Im|']:.6f}")
    print(f"Std  |Im(offdiag)|     : {results['outer']['std_|Im|']:.6f}")
    print(f"Coefficient of Variation: {results['outer']['coeff_of_variation']:.4f}")

    print("\n--- Comparison ---")
    print(f"CV Ratio (Central / Outer): {results['cv_ratio_central_over_outer']:.4f}")

    print("\n[Analysis complete]")