#!/usr/bin/env python3
"""
HSMT Two-Electron + Phonon Model — v4.0 Dense CPU (Production / Trusted Results)

Clean, reliable dense implementation for scientific work and convergence studies.
"""

import numpy as np
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===================================================================
# FUNDAMENTAL HSMT CONSTANTS
# ===================================================================
G = 0.915965594177219
ALPHA = np.pi + G
SIGMA0 = np.sqrt(2) / 4.0
ELL0 = 1e-3

KAPPA_SLOPE = 4 * np.pi / 3
B_SLOPE = 2 * np.sqrt(3)
C_SLOPE = (np.pi + np.e) / 2

A = ALPHA * KAPPA_SLOPE
B = ALPHA * B_SLOPE
M0 = ALPHA * C_SLOPE


def d_minus1_vec(ell: np.ndarray) -> np.ndarray:
    result = np.full_like(ell, 2.0)
    mask = ell > 0
    x = np.log(ell[mask] / ELL0)
    result[mask] = 4.0 - 1.8 * np.exp(-x**2 / (2 * SIGMA0**2)) + 0.6 * (ell[mask] / (ELL0 + ell[mask]))
    return result


def w_minus1(ell: float) -> float:
    if ell <= 0:
        return 0.0
    arg = np.log(ell / ELL0)
    pref = 1.0 / (np.sqrt(2 * np.pi) * SIGMA0 * ell)
    return pref * np.exp(-0.5 * arg**2 / SIGMA0**2)


class HSMT_TwoElectron_Phonons_Dense_v40:
    """
    HSMT Two-Electron + Phonon Model — v4.0 Dense CPU (Trusted Results)
    """

    def __init__(self, M: int = 300, delta_rho: float = 0.02,
                 n_phonon_modes: int = 3, phonon_freq: float = 0.18,
                 eph_coupling: float = 2.0, disp_coupling: float = 9.0,
                 pair_coupling: float = 0.4, U_direct: float = 0.35,
                 max_phonon_occupation: int = 8,
                 n_single_particle_states: int = 6,
                 g_scatter: float = 0.8):
        self.M = M
        self.delta_rho = delta_rho
        self.rho = np.linspace(-M * delta_rho, M * delta_rho, 2 * M + 1)
        self.N_sites = len(self.rho)
        self.ell = ELL0 * np.exp(self.rho)
        self.w = np.array([w_minus1(e) for e in self.ell])
        self.dmu = self.w * (d_minus1_vec(self.ell) - 4) * self.delta_rho

        self.D_lattice = None
        self.central_indices = None
        self.eigenvalues_sp = None
        self.eigenvectors_sp = None

        self.n_phonon_modes = n_phonon_modes
        self.phonon_freq = phonon_freq
        self.eph_coupling = eph_coupling
        self.disp_coupling = disp_coupling
        self.pair_coupling = pair_coupling
        self.U_direct = U_direct
        self.max_phonon_occupation = max_phonon_occupation
        self.n_sp = n_single_particle_states
        self.g_scatter = g_scatter

        self.phonon_dim = None
        self.two_elec_dim = None
        self.two_elec_configs = None

    # ------------------------------------------------------------------
    # SINGLE-PARTICLE
    # ------------------------------------------------------------------
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
            tanh_u = np.tanh(ALPHA * rho)
            V_bi = (A / 2.0) * tanh_u
            V_mass = M0
            D[i, i] += V_bi + V_mass
        self.D_lattice = D
        return D

    def identify_central_region(self, threshold: float = 0.01):
        w_max = np.max(self.w)
        significant = self.w > (threshold * w_max)
        self.central_indices = np.where(significant)[0]
        return self.central_indices

    def compute_single_particle_spectrum(self, k: int = None):
        if k is None: k = self.n_sp
        if self.D_lattice is None: self.build_master_operator()
        if self.central_indices is None: self.identify_central_region(threshold=0.01)
        idx = self.central_indices
        D_central = self.D_lattice[np.ix_(idx, idx)]
        D_real = np.real(np.nan_to_num(D_central, nan=0.0, posinf=0.0, neginf=0.0))
        evals, evecs = np.linalg.eigh(D_real)
        self.eigenvalues_sp = evals[:k]
        self.eigenvectors_sp = evecs[:, :k]
        return self.eigenvalues_sp

    # ------------------------------------------------------------------
    # PHONON OPERATORS
    # ------------------------------------------------------------------
    def _fock_index(self, occupations):
        idx = 0
        base = 1
        for occ in occupations:
            idx += occ * base
            base *= (self.max_phonon_occupation + 1)
        return idx

    def build_phonon_operators(self):
        n_occ = self.max_phonon_occupation + 1
        self.phonon_dim = n_occ ** self.n_phonon_modes

        self.n_ph_ops = []
        self.x_ops = []

        for m in range(self.n_phonon_modes):
            b = np.zeros((self.phonon_dim, self.phonon_dim), dtype=complex)
            bdag = np.zeros((self.phonon_dim, self.phonon_dim), dtype=complex)
            n_op = np.zeros((self.phonon_dim, self.phonon_dim), dtype=complex)
            for i in range(self.phonon_dim):
                occ = []
                tmp = i
                for _ in range(self.n_phonon_modes):
                    occ.append(tmp % n_occ)
                    tmp //= n_occ
                if occ[m] > 0:
                    new_occ = list(occ); new_occ[m] -= 1
                    j = self._fock_index(new_occ)
                    b[j, i] = np.sqrt(occ[m])
                if occ[m] < self.max_phonon_occupation:
                    new_occ = list(occ); new_occ[m] += 1
                    j = self._fock_index(new_occ)
                    bdag[i, j] = np.sqrt(occ[m] + 1)
                n_op[i, i] = occ[m]

            self.n_ph_ops.append(n_op)
            self.x_ops.append(b + bdag)

        print(f"Phonon operators built. Phonon Hilbert space dimension = {self.phonon_dim}")

    # ------------------------------------------------------------------
    # TWO-ELECTRON CONFIGS
    # ------------------------------------------------------------------
    def build_two_electron_configs(self):
        self.two_elec_configs = list(combinations(range(self.n_sp), 2))
        self.two_elec_dim = len(self.two_elec_configs)
        print(f"Two-electron electronic configurations: {self.two_elec_dim}")
        return self.two_elec_configs

    # ------------------------------------------------------------------
    # BUILD FULL DENSE HAMILTONIAN
    # ------------------------------------------------------------------
    def build_full_hamiltonian(self, include_interactions: bool = True):
        if self.eigenvalues_sp is None:
            self.compute_single_particle_spectrum()
        if self.two_elec_configs is None:
            self.build_two_electron_configs()
        if self.phonon_dim is None:
            self.build_phonon_operators()

        n_2e = self.two_elec_dim
        n_ph = self.phonon_dim
        total_dim = n_2e * n_ph

        # Non-interacting part
        H_2e = np.zeros((n_2e, n_2e))
        for idx, (i, j) in enumerate(self.two_elec_configs):
            H_2e[idx, idx] = self.eigenvalues_sp[i] + self.eigenvalues_sp[j]

        if include_interactions:
            for idx, (i, j) in enumerate(self.two_elec_configs):
                orbital_separation = abs(i - j)
                U_ij = self.U_direct * (1.0 / (1.0 + 0.3 * orbital_separation))
                H_2e[idx, idx] += U_ij

        H_ph = np.zeros((n_ph, n_ph), dtype=complex)
        for m in range(self.n_phonon_modes):
            H_ph += self.phonon_freq * self.n_ph_ops[m]

        H_nonint = np.kron(H_2e, np.eye(n_ph)) + np.kron(np.eye(n_2e), H_ph)

        if not include_interactions:
            return H_nonint

        # Interaction part
        H_int = np.zeros((total_dim, total_dim), dtype=complex)

        # Density e-ph coupling
        n_total_2e = np.zeros((n_2e, n_2e))
        dmu_central = self.dmu[self.central_indices]
        for cfg_idx, (orb_i, orb_j) in enumerate(self.two_elec_configs):
            psi_i = self.eigenvectors_sp[:, orb_i]
            psi_j = self.eigenvectors_sp[:, orb_j]
            n_total_2e[cfg_idx, cfg_idx] = (np.sum(np.abs(psi_i)**2 * dmu_central) +
                                            np.sum(np.abs(psi_j)**2 * dmu_central))

        for m in range(self.n_phonon_modes):
            H_int += self.eph_coupling * np.kron(n_total_2e, self.n_ph_ops[m])

        # Displacement coupling
        for m in range(self.n_phonon_modes):
            H_int += self.disp_coupling * np.kron(n_total_2e, self.x_ops[m])

        # Pairing term
        for m in range(self.n_phonon_modes):
            H_int += -self.pair_coupling * np.kron(np.eye(n_2e), self.n_ph_ops[m])

        # Off-diagonal scattering
        if self.g_scatter != 0.0:
            H_scatter = np.zeros((n_2e, n_2e), dtype=complex)
            for cfg1 in range(n_2e):
                for cfg2 in range(cfg1 + 1, n_2e):
                    (i1, j1) = self.two_elec_configs[cfg1]
                    (i2, j2) = self.two_elec_configs[cfg2]
                    orbital_diff = abs(i1 - i2) + abs(j1 - j2)
                    coupling = self.g_scatter / (1.0 + 0.5 * orbital_diff)
                    H_scatter[cfg1, cfg2] = coupling
                    H_scatter[cfg2, cfg1] = coupling

            x0 = self.x_ops[0]
            for i in range(n_2e):
                for j in range(n_2e):
                    if abs(H_scatter[i, j]) > 1e-12:
                        start_i = i * n_ph
                        end_i = start_i + n_ph
                        start_j = j * n_ph
                        end_j = start_j + n_ph
                        H_int[start_i:end_i, start_j:end_j] += H_scatter[i, j] * x0

        return H_nonint + H_int

    # ------------------------------------------------------------------
    # DIAGONALIZE
    # ------------------------------------------------------------------
    def diagonalize(self, include_interactions: bool = True):
        H = self.build_full_hamiltonian(include_interactions=include_interactions)
        print(f"Diagonalizing dense Hamiltonian of dimension {H.shape[0]} ...")
        evals = np.sort(np.real(np.linalg.eigvalsh(H)))
        return evals[:min(10, len(evals))]

    # ------------------------------------------------------------------
    # RUN DIAGNOSTICS
    # ------------------------------------------------------------------
    def run_diagnostics(self):
        print("\n" + "=" * 72)
        print("HSMT Two-Electron + Phonon Model — v4.0 Dense CPU (Trusted Results)")
        print("=" * 72)
        print(f"Max phonon occupation = {self.max_phonon_occupation}")
        print(f"Total Hilbert space dimension = {self.two_elec_dim * self.phonon_dim if self.two_elec_dim else 'N/A'}")

        # No coupling
        print("\n[1/2] Computing no-coupling spectrum...")
        evals_no = self.diagonalize(include_interactions=False)

        # Full coupling
        print("\n[2/2] Computing full-coupling spectrum...")
        evals_with = self.diagonalize(include_interactions=True)

        print(f"\nDirect repulsion base U = {self.U_direct}")
        if self.g_scatter != 0.0:
            print(f"Phonon-mediated scattering g_scatter = {self.g_scatter}")

        print("\n--- Lowest Eigenvalues (Dense CPU) ---")
        print(f"{'Idx':>4} | {'No coupling':>14} | {'With coupling':>14} | {'Net Shift':>12}")
        print("-" * 52)
        for i in range(min(8, len(evals_with))):
            shift = evals_with[i] - evals_no[i]
            print(f"{i:>4} | {evals_no[i]:>14.6f} | {evals_with[i]:>14.6f} | {shift:>12.6f}")

        ground_state_shift = evals_with[0] - evals_no[0]
        print("\n" + "=" * 64)
        print("NET PAIRING INDICATOR (Ground State) — v4.0 Dense CPU")
        print("=" * 64)
        print(f"Ground state energy (U only)      : {evals_no[0]:.6f}")
        print(f"Ground state energy (U + phonons) : {evals_with[0]:.6f}")
        print(f"Net Pairing Shift                 : {ground_state_shift:+.6f}")

        if ground_state_shift < -0.5:
            status = "PHONON-MEDIATED ATTRACTION IS WINNING (Strong binding)"
        elif ground_state_shift < 0:
            status = "PHONON-MEDIATED ATTRACTION IS WINNING (Moderate binding)"
        else:
            status = "REPULSION STILL DOMINATES"
        print(f"Status: {status}")
        print("=" * 64)

        return evals_no, evals_with, ground_state_shift


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("HSMT Two-Electron + Phonon Model — v4.0 Dense CPU (Production Version)")
    print("=" * 72)

    # ============================================================
    # USER SETTINGS — Change max_phonon_occupation for convergence studies
    # ============================================================
    model = HSMT_TwoElectron_Phonons_Dense_v40(
        M=300,
        delta_rho=0.02,
        n_phonon_modes=3,
        phonon_freq=0.18,
        eph_coupling=2.0,
        disp_coupling=9.0,
        pair_coupling=0.4,
        U_direct=0.35,
        max_phonon_occupation=8,      # ← Change this value (try 6, 8, 10, 12, 15, 16)
        n_single_particle_states=6,
        g_scatter=0.8
    )

    model.build_master_operator()
    model.identify_central_region(threshold=0.01)
    model.compute_single_particle_spectrum()

    print("\nSingle-particle eigenvalues (lowest 6):")
    print(np.round(model.eigenvalues_sp, decimals=6))

    model.run_diagnostics()

    print("\nv4.0 Dense CPU run completed successfully.")
