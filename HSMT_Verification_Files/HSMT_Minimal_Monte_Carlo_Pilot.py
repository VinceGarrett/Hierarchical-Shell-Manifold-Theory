#!/usr/bin/env python3
"""
HSMT Phase 5 – Monte Carlo at M=1500
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

class HSMT_Disorder_MonteCarlo_v3:
    def __init__(self, M, physical_half_length=6.0, beta=5.0, 
                 target_mean_G=0.60, target_Delta=4.80,
                 action_strength_G=90.0, action_strength_Delta=25.0,
                 disorder_strength=0.50, n_eigen_factor=0.45,
                 delta_min=4.2, delta_max=5.5):
        self.M = M
        self.physical_half_length = physical_half_length
        self.delta_rho = physical_half_length / M
        self.rho = np.linspace(-physical_half_length, physical_half_length, 2 * M + 1)
        self.N = len(self.rho)
        self.beta = beta
        self.target_mean_G = target_mean_G
        self.target_Delta = target_Delta
        self.action_strength_G = action_strength_G
        self.action_strength_Delta = action_strength_Delta
        self.disorder_strength = disorder_strength
        self.n_eigen_factor = n_eigen_factor
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.current_disorder = np.zeros(self.N)

    def build_operator(self, disorder):
        h = self.delta_rho
        coeff = 1j / (12 * h)
        main = np.zeros(self.N, dtype=complex)
        off1 = np.full(self.N-1,  8*coeff, dtype=complex)
        off2 = np.full(self.N-2, -coeff,   dtype=complex)

        D_fd = diags(
            diagonals=[main, off1, off1.conjugate(), off2, off2.conjugate()],
            offsets=[0, 1, -1, 2, -2],
            format='csr'
        )

        V = np.zeros(self.N, dtype=complex)
        for i in range(self.N):
            r = self.rho[i]
            tanh_u = np.tanh(np.pi * r)
            V_bi = (np.pi + 0.915965594177219) * 2 * np.pi / 3 * tanh_u / 2.0
            V_mass = (np.pi + 0.915965594177219) * (np.pi + np.e) / 2
            V[i] = V_bi + V_mass + 1j * disorder[i]

        return D_fd + diags([V], offsets=[0], format='csr')

    def evaluate_G(self, disorder):
        D = self.build_operator(disorder)
        n_eigen = max(500, int(self.M * self.n_eigen_factor))
        eigvals, eigvecs = eigs(D, k=n_eigen, which='SR', maxiter=200000, tol=1e-10)

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
            return None

        rho_min = self.rho[top_idx[0]]
        rho_max = self.rho[top_idx[-1]]
        Delta_rho = rho_max - rho_min

        if not (self.delta_min <= Delta_rho <= self.delta_max):
            return None

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
        return {"G": G, "Delta_rho": Delta_rho}

    def effective_action(self, G, Delta_rho):
        action_G = self.action_strength_G * (G - self.target_mean_G)**2
        action_Delta = self.action_strength_Delta * (Delta_rho - self.target_Delta)**2
        return action_G + action_Delta

    def propose_disorder(self, current, step_size):
        return current + np.random.normal(0.0, step_size, self.N)

    def metropolis_step(self, current_disorder, step_size):
        proposal = self.propose_disorder(current_disorder, step_size)
        geo = self.evaluate_G(proposal)
        if geo is None:
            return current_disorder, False, None

        current_action = self.effective_action(self.current_G, self.current_Delta)
        proposal_action = self.effective_action(geo["G"], geo["Delta_rho"])
        delta_S = proposal_action - current_action
        accept_prob = min(1.0, np.exp(-self.beta * delta_S))

        if np.random.rand() < accept_prob:
            self.current_G = geo["G"]
            self.current_Delta = geo["Delta_rho"]
            return proposal, True, geo
        else:
            return current_disorder, False, geo

    def run_chain(self, n_steps=140, step_size=0.07, burn_in=50, print_every=15):
        self.current_disorder = np.random.normal(0.0, self.disorder_strength, self.N)
        geo0 = self.evaluate_G(self.current_disorder)
        self.current_G = geo0["G"] if geo0 else 0.8
        self.current_Delta = geo0["Delta_rho"] if geo0 else 4.8

        accepted = 0
        trace_G = []
        trace_Delta = []

        print(f"{'Step':>6} | {'mean|imag|':>11} | {'G':>9} | {'Δρ':>6} | Accept")
        print("-" * 55)

        for step in range(n_steps):
            new_dis, accepted_step, geo = self.metropolis_step(self.current_disorder, step_size)
            if accepted_step:
                self.current_disorder = new_dis
                accepted += 1

            if step >= burn_in and geo is not None:
                trace_G.append(self.current_G)
                trace_Delta.append(self.current_Delta)

            if step % print_every == 0 or step < 6:
                acc_str = "Yes" if accepted_step else "No"
                dR = geo["Delta_rho"] if geo else np.nan
                print(f"{step:6d} | {np.mean(np.abs(self.current_disorder)):11.4f} | {self.current_G:9.4f} | {dR:6.2f} | {acc_str}")

        return {
            "acceptance_rate": accepted / n_steps,
            "mean_G": np.nanmean(trace_G),
            "std_G": np.nanstd(trace_G),
            "mean_Delta": np.nanmean(trace_Delta),
            "n_samples": len(trace_G)
        }


if __name__ == "__main__":
    print("HSMT Phase 5 – Monte Carlo at M=1500")
    mc = HSMT_Disorder_MonteCarlo_v3(
        M=1500, beta=5.0, target_mean_G=0.60, target_Delta=4.80,
        action_strength_G=90.0, action_strength_Delta=25.0, disorder_strength=0.50
    )
    result = mc.run_chain(n_steps=140, step_size=0.07, burn_in=50, print_every=15)

    print("\n" + "=" * 55)
    print(f"Acceptance rate      : {result['acceptance_rate']:.3f}")
    print(f"Mean G (post burn-in): {result['mean_G']:.4f} ± {result['std_G']:.4f}")
    print(f"Mean Δρ              : {result['mean_Delta']:.2f}")
    print(f"Number of samples    : {result['n_samples']}")