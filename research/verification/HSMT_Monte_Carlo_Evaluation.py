import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import njit
from scipy.sparse.linalg import LinearOperator, eigs

# ============================================================
# Matrix-Free Operator
# ============================================================
@njit(fastmath=True, cache=True)
def apply_derivative_numba(psi, drho):
    M = psi.shape[0]
    dpsi = np.zeros_like(psi)
    for i in range(2, M-2):
        dpsi[i] = (-psi[i+2] + 8*psi[i+1] - 8*psi[i-1] + psi[i-2]) / (12*drho)
    dpsi[0] = (psi[1] - psi[0]) / drho
    dpsi[1] = (psi[2] - psi[0]) / (2*drho)
    dpsi[M-2] = (psi[M-1] - psi[M-3]) / (2*drho)
    dpsi[M-1] = (psi[M-1] - psi[M-2]) / drho
    return dpsi

@njit(fastmath=True, cache=True)
def apply_octonion_bimultiplication_numba(psi1, psi2, u_real, u_imag):
    M = psi1.shape[0]
    out1 = np.zeros_like(psi1)
    out2 = np.zeros_like(psi2)
    for i in range(M):
        dot_L = np.sum(u_imag[i] * psi2[i])
        dot_R = np.sum(u_imag[i] * psi2[i])
        L_psi1 = u_real[i] * psi1[i] - dot_L
        L_psi2 = u_real[i] * psi2[i] + np.sum(u_imag[i] * psi1[i])
        R_psi1 = u_real[i] * psi1[i] + dot_R
        R_psi2 = u_real[i] * psi2[i] - np.sum(u_imag[i] * psi1[i])
        out1[i] = 0.5 * (L_psi1 + R_psi1)
        out2[i] = 0.5 * (L_psi2 + R_psi2)
    return out1, out2

class D_lattice_matrix_free(LinearOperator):
    def __init__(self, rho, Delta_mu, u_rho, params):
        self.rho = rho
        self.Delta_mu = Delta_mu
        self.u_rho = u_rho.copy()
        self.u_real = u_rho[:, 0]
        self.u_imag = u_rho[:, 1:]
        self.params = params
        self.drho = rho[1] - rho[0]
        self.M = len(rho)
        self.shape = (2 * self.M, 2 * self.M)
        self.dtype = np.dtype('float64')

    def _matvec(self, x):
        x = np.asarray(x).ravel()
        psi1 = x[0::2]
        psi2 = x[1::2]
        dpsi1 = apply_derivative_numba(psi1, self.drho)
        dpsi2 = apply_derivative_numba(psi2, self.drho)
        oct1, oct2 = apply_octonion_bimultiplication_numba(
            psi1, psi2, self.u_real, self.u_imag
        )
        m0 = self.params['m0']
        res1 = dpsi1 + oct2 - m0 * psi2
        res2 = dpsi2 - oct1 + m0 * psi1
        result = np.empty_like(x)
        result[0::2] = res1
        result[1::2] = res2
        return result

    def norm(self, psi):
        return np.sqrt(np.sum(np.abs(psi)**2 * self.Delta_mu.repeat(2)))

# ============================================================
# Standalone Parallel Worker (Tuned for M=2500)
# ============================================================
def _compute_g_worker(rho, Delta_mu, base_u, disorder_strength, params, k_eigen):
    u_dis = base_u + np.random.normal(0.0, disorder_strength, size=base_u.shape)
    D = D_lattice_matrix_free(rho, Delta_mu, u_dis, params)

    try:
        evals, evecs = eigs(
            D, k=k_eigen, which='SR',
            maxiter=150,
            tol=1.5e-3,
            ncv=45
        )
        real_parts = np.abs(evals.real)
        best_idx = np.argmin(real_parts)

        if real_parts[best_idx] > 4.8:
            return np.nan

        psi = evecs[:, best_idx]
        ipr = np.sum(np.abs(psi)**4)
        return 1.0 / ipr if ipr > 1e-12 else np.nan
    except Exception:
        return np.nan

# ============================================================
# Monte Carlo Class
# ============================================================
class MonteCarloG:
    def __init__(self, D_operator, n_samples=500, disorder_strength=0.12, 
                 k_eigen=3, n_jobs=-1, seed=None):
        self.D = D_operator
        self.M = D_operator.M
        self.n_samples = n_samples
        self.disorder_strength = disorder_strength
        self.k_eigen = k_eigen
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(seed)
        self.g_values = np.array([])

    def run(self):
        print(f"Running {self.n_samples} samples using {self.n_jobs} parallel workers...")
        base_u = self.D.u_rho.copy()
        rho = self.D.rho
        Delta_mu = self.D.Delta_mu
        params = self.D.params

        self.g_values = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_g_worker)(
                rho, Delta_mu, base_u, self.disorder_strength, params, self.k_eigen
            )
            for _ in range(self.n_samples)
        )
        self.g_values = np.array(self.g_values)

        valid = ~np.isnan(self.g_values)
        print(f"\nMonte Carlo finished.")
        print(f"Valid samples: {np.sum(valid)} / {self.n_samples} "
              f"({100 * np.sum(valid) / self.n_samples:.1f}%)")

    def get_statistics(self):
        valid = ~np.isnan(self.g_values)
        if np.sum(valid) == 0:
            return None
        g_valid = self.g_values[valid]
        return {
            'n_samples': self.n_samples,
            'n_valid': int(np.sum(valid)),
            'validity_rate': np.sum(valid) / self.n_samples,
            'mean': float(np.mean(g_valid)),
            'median': float(np.median(g_valid)),
            'std': float(np.std(g_valid)),
            'min': float(np.min(g_valid)),
            'max': float(np.max(g_valid)),
            'p25': float(np.percentile(g_valid, 25)),
            'p75': float(np.percentile(g_valid, 75))
        }

    def plot_histogram(self, bins=50, save_path="G_histogram_M2500.png"):
        valid = ~np.isnan(self.g_values)
        g_valid = self.g_values[valid]
        plt.figure(figsize=(9, 5))
        plt.hist(g_valid, bins=bins, color='steelblue', edgecolor='black', alpha=0.75)
        plt.axvline(np.mean(g_valid), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(g_valid):.2f}')
        plt.axvline(np.median(g_valid), color='green', linestyle='-.', linewidth=2,
                    label=f'Median = {np.median(g_valid):.2f}')
        plt.xlabel('Geometric Criterion G')
        plt.ylabel('Count')
        plt.title(f'Distribution of G at M={self.M} (Disorder = {self.disorder_strength})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")

    def save_results(self, filename="G_results_M2500_n800.npy"):
        np.save(filename, self.g_values)
        print(f"Full G values saved to {filename}")

    def scan_disorder_strength(self, strengths=None, n_samples_per_point=150):
        if strengths is None:
            strengths = np.linspace(0.05, 0.25, 5)
        results = []
        for s in strengths:
            self.disorder_strength = s
            self.n_samples = n_samples_per_point
            self.run()
            stats = self.get_statistics()
            if stats:
                results.append({
                    'disorder_strength': s,
                    'mean_G': stats['mean'],
                    'std_G': stats['std'],
                    'validity': stats['validity_rate']
                })
                print(f"Disorder = {s:.3f} → Mean G = {stats['mean']:.2f} ± {stats['std']:.2f} "
                      f"(validity = {stats['validity_rate']*100:.1f}%)")
        if len(results) > 1:
            plt.figure(figsize=(7, 4))
            plt.errorbar([r['disorder_strength'] for r in results],
                         [r['mean_G'] for r in results],
                         yerr=[r['std_G'] for r in results],
                         fmt='o-', capsize=4, color='darkblue')
            plt.xlabel('Disorder Strength')
            plt.ylabel('Mean G')
            plt.title('G vs Disorder Strength (M=2500)')
            plt.grid(True, alpha=0.3)
            plt.savefig("G_vs_disorder_M2500.png", dpi=150, bbox_inches='tight')
            print("Scan plot saved to G_vs_disorder_M2500.png")
        return results

# ============================================================
# Main - Light-to-Moderate Statistics at M=2500
# ============================================================
if __name__ == "__main__":
    # Lattice
    M = 2500
    rho = np.linspace(-7.5, 7.5, M)
    drho = rho[1] - rho[0]
    Delta_mu = np.ones(M) * drho
    u_rho = np.zeros((M, 8))
    u_rho[:, 0] = np.tanh(rho)
    u_rho[:, 1:] = 0.32 * np.tanh(rho)[:, np.newaxis]
    params = {'m0': 1.0}
    D_op = D_lattice_matrix_free(rho, Delta_mu, u_rho, params)

    # Light-to-moderate statistics at M=2500
    mc = MonteCarloG(D_op, n_samples=800, disorder_strength=0.12, n_jobs=-1, seed=42)
    mc.run()

    # Analysis
    stats = mc.get_statistics()
    print("\n=== Results (M=2500) ===")
    for k, v in stats.items():
        print(f"{k:18}: {v}")

    mc.plot_histogram(bins=50)
    mc.save_results("G_results_M2500_n800.npy")

    print("\n=== Disorder Strength Scan ===")
    mc.scan_disorder_strength(n_samples_per_point=150)