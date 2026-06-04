#!/usr/bin/env python3
"""
HSMT Phase 3 – Intrinsic Central Region: Sensitivity to Cutoff Multiple
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HSMT_Phase3_Intrinsic_Sensitivity:
    def __init__(self, M=300, delta_rho=0.02):
        self.M = M
        self.delta_rho = delta_rho
        self.rho = np.linspace(-M * delta_rho, M * delta_rho, 2 * M + 1)
        self.N_sites = len(self.rho)
        self.D_lattice = None

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

    def compute_for_multiple(self, multiple):
        D_real = np.real(self.D_lattice)
        diag_real = np.abs(np.diag(D_real))
        global_mean = np.mean(diag_real)
        global_std = np.std(diag_real)

        # Intrinsic definition with variable multiple
        significant = np.abs(diag_real - global_mean) <= (multiple * global_std)
        idx = np.where(significant)[0]

        if len(idx) < 5:
            return None

        rho_min = self.rho[idx[0]]
        rho_max = self.rho[idx[-1]]
        Delta_rho = rho_max - rho_min

        diag_real_c = diag_real[idx]
        mean_real = np.mean(diag_real_c)
        std_real = np.std(diag_real_c)
        H = mean_real / std_real if std_real > 0 else np.nan

        D_imag = np.imag(self.D_lattice)
        D_imag_block = D_imag[np.ix_(idx, idx)]
        offdiag_mask = ~np.eye(len(idx), dtype=bool)
        imag_vals = np.abs(D_imag_block[offdiag_mask])
        mean_imag = np.mean(imag_vals)
        std_imag = np.std(imag_vals)
        C = mean_imag / std_imag if std_imag > 0 else np.nan

        G = (Delta_rho * H * C) / mean_imag if mean_imag > 0 else np.nan
        G_prime = (Delta_rho * (H ** 1.5) * C) / mean_imag if mean_imag > 0 else np.nan

        return {
            "multiple": multiple,
            "num_sites": len(idx),
            "Delta_rho": Delta_rho,
            "H": H,
            "C": C,
            "mean_imag": mean_imag,
            "G": G,
            "G_prime": G_prime
        }


# ============================================================
# Run Sensitivity Analysis
# ============================================================
if __name__ == "__main__":
    print("=" * 82)
    print("HSMT Phase 3 – Intrinsic Central Region: Sensitivity to Cutoff Multiple")
    print("=" * 82)

    model = HSMT_Phase3_Intrinsic_Sensitivity(M=300, delta_rho=0.02)
    model.build_master_operator()

    multiples = [0.8, 1.0, 1.2]

    print(f"\n{'Multiple':>8} | {'Sites':>6} | {'Δρ':>6} | {'H':>7} | {'C':>7} | {'G':>8} | {'G\'':>8}")
    print("-" * 65)

    for mult in multiples:
        res = model.compute_for_multiple(mult)
        if res is not None:
            print(f"{res['multiple']:8.1f} | {res['num_sites']:6d} | {res['Delta_rho']:6.2f} | "
                  f"{res['H']:7.4f} | {res['C']:7.4f} | {res['G']:8.4f} | {res['G_prime']:8.4f}")

    print("\n[Analysis complete]")