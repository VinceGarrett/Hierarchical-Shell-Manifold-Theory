#!/usr/bin/env python3
"""
HSMT Phase 3 – Full Complex Spectral Definition of Central Region
Locates near-degenerate cluster in the complex plane
"""

import numpy as np
from scipy.linalg import eig
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HSMT_Phase3_FullComplex_Spectral_Region:
    def __init__(self, M=300, delta_rho=0.02, window_size=6, weight_threshold=0.80):
        self.M = M
        self.delta_rho = delta_rho
        self.rho = np.linspace(-M * delta_rho, M * delta_rho, 2 * M + 1)
        self.N_sites = len(self.rho)
        self.D_lattice = None
        self.window_size = window_size
        self.weight_threshold = weight_threshold
        self.central_indices = None
        self.cluster_eigvals = None

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

    def find_near_degenerate_cluster(self):
        # Use full complex operator
        eigvals, eigvecs = eig(self.D_lattice)

        # Sort by real part for sliding window
        sort_idx = np.argsort(np.real(eigvals))
        eigvals_sorted = eigvals[sort_idx]
        eigvecs_sorted = eigvecs[:, sort_idx]

        min_diameter = np.inf
        best_start = 0

        for i in range(len(eigvals_sorted) - self.window_size + 1):
            window = eigvals_sorted[i : i + self.window_size]
            # Diameter of the set in complex plane (max pairwise |λ_i - λ_j|)
            diffs = np.abs(window[:, np.newaxis] - window[np.newaxis, :])
            diameter = np.max(diffs)
            if diameter < min_diameter:
                min_diameter = diameter
                best_start = i

        cluster_idx = np.arange(best_start, best_start + self.window_size)
        self.cluster_eigvals = eigvals_sorted[cluster_idx]

        # Compute total weight per site from right eigenvectors of this cluster
        weights = np.sum(np.abs(eigvecs_sorted[:, cluster_idx])**2, axis=1)

        # Select sites carrying the top weight_threshold of total weight
        total_weight = np.sum(weights)
        sorted_weights = np.sort(weights)[::-1]
        cumulative = np.cumsum(sorted_weights) / total_weight
        cutoff_rank = np.searchsorted(cumulative, self.weight_threshold) + 1

        top_indices = np.argsort(weights)[::-1][:cutoff_rank]
        self.central_indices = np.sort(top_indices)

        return self.central_indices, self.cluster_eigvals, min_diameter

    def compute_geometric_quantities(self):
        if self.central_indices is None:
            self.find_near_degenerate_cluster()

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

        G = (Delta_rho * H * C) / mean_imag if mean_imag > 0 else np.nan
        G_prime = (Delta_rho * (H ** 1.5) * C) / mean_imag if mean_imag > 0 else np.nan

        return {
            "num_sites": len(idx),
            "Delta_rho": Delta_rho,
            "H": H,
            "C": C,
            "mean_imag": mean_imag,
            "cluster_diameter": min_diameter,  # from find_near_degenerate_cluster
            "G": G,
            "G_prime": G_prime
        }


# ============================================================
# Run Full Complex Spectral Definition
# ============================================================
if __name__ == "__main__":
    print("=" * 82)
    print("HSMT Phase 3 – Full Complex Spectral Definition of Central Region")
    print("=" * 82)

    model = HSMT_Phase3_FullComplex_Spectral_Region(M=300, delta_rho=0.02,
                                                     window_size=6, weight_threshold=0.80)
    model.build_master_operator()
    indices, cluster_eigvals, min_diameter = model.find_near_degenerate_cluster()
    results = model.compute_geometric_quantities()

    print(f"\nNear-degenerate cluster diameter : {min_diameter:.6e}")
    print(f"Spectral central region sites    : {results['num_sites']}")
    print(f"Central ρ-width (Δρ)             : {results['Delta_rho']:.2f}")
    print(f"Homogeneity factor (H)           : {results['H']:.4f}")
    print(f"Coherence factor (C)             : {results['C']:.4f}")
    print(f"Mean |Im(offdiag)|               : {results['mean_imag']:.6f}")
    print(f"\nOriginal criterion G             : {results['G']:.4f}")
    print(f"Refined criterion G' (H^1.5)     : {results['G_prime']:.4f}")

    print("\n[Analysis complete]")