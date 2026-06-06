import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ============================================================
# Load All Six Data Files
# ============================================================
files = {
    800:  "G_results_M800_n1500.npy",
    1000: "G_results_M1000_n1500.npy",
    1200: "G_results_M1200_n1500.npy",
    1500: "G_results_M1500_n1500.npy",
    2000: "G_results_M2000_n1000.npy",
    2500: "G_results_M2500_n800.npy"
}

data = {}
for m, fname in files.items():
    g = np.load(fname, allow_pickle=True)
    data[m] = g[~np.isnan(g)]

Ms = np.array(list(data.keys()))
invM = 1.0 / Ms

# ============================================================
# Compute Statistics for Each Size
# ============================================================
def compute_stats(d):
    return {
        'mean': np.mean(d),
        'median': np.median(d),
        'std': np.std(d),
        'skewness': stats.skew(d) if len(d) > 1 else np.nan,
        'kurtosis': stats.kurtosis(d) if len(d) > 1 else np.nan,
        'p90': np.percentile(d, 90),
        'p95': np.percentile(d, 95),
        'p99': np.percentile(d, 99),
    }

from scipy import stats
stats_all = {m: compute_stats(data[m]) for m in Ms}

# ============================================================
# Linear Extrapolation (M → ∞)
# ============================================================
def extrapolate(y):
    slope, intercept, r_value, _, _ = linregress(invM, y)
    return intercept, r_value**2

extrapolated = {}
for key in ['mean', 'median', 'skewness', 'kurtosis', 'p90', 'p95', 'p99']:
    y = np.array([stats_all[m][key] for m in Ms])
    cont, r2 = extrapolate(y)
    extrapolated[key] = {'value': cont, 'R2': r2}

# ============================================================
# Print Summary Table
# ============================================================
print("\n" + "="*85)
print("FINITE-SIZE SCALING SUMMARY — Hierarchical Shell-Manifold Theory (HSMT)")
print("="*85)
print(f"{'Metric':<12}", end="")
for m in Ms:
    print(f"M={m:>6}", end="   ")
print("Continuum   R²")
print("-"*85)

for key in ['mean', 'median', 'skewness', 'kurtosis', 'p90', 'p95', 'p99']:
    print(f"{key:<12}", end="")
    for m in Ms:
        print(f"{stats_all[m][key]:>8.2f}", end="   ")
    print(f"{extrapolated[key]['value']:>8.2f}   {extrapolated[key]['R2']:.3f}")

print("="*85)
print("\nNote: Continuum values from linear extrapolation in 1/M.")

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(Ms)))

# Overlaid Histograms
ax1 = axes[0, 0]
for i, m in enumerate(Ms):
    ax1.hist(data[m], bins=60, density=True, alpha=0.4, color=colors[i], label=f'M={m}')
ax1.set_xlabel('Geometric Criterion G')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of G across System Sizes')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Mean & Median vs M
ax2 = axes[0, 1]
means = [stats_all[m]['mean'] for m in Ms]
medians = [stats_all[m]['median'] for m in Ms]
ax2.plot(Ms, means, 'o-', color='blue', markersize=8, label='Mean G')
ax2.plot(Ms, medians, 's--', color='red', markersize=8, label='Median G')
ax2.set_xlabel('Lattice Size M')
ax2.set_ylabel('G')
ax2.set_title('Mean and Median G vs System Size')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Tail Comparison P(G > t)
ax3 = axes[1, 0]
thresholds = np.linspace(50, 700, 150)
for i, m in enumerate(Ms):
    tail = [np.sum(data[m] > t) / len(data[m]) for t in thresholds]
    ax3.plot(thresholds, tail, label=f'M={m}', color=colors[i])
ax3.set_xlabel('G Threshold')
ax3.set_ylabel('Fraction of Samples > Threshold')
ax3.set_title('Tail Comparison P(G > t)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Extrapolation of Mean G
ax4 = axes[1, 1]
ax4.scatter(invM, means, color='black', s=70, zorder=5, label='Data')
slope, intercept, r_value, _, _ = linregress(invM, means)
x_fit = np.linspace(0, invM.max()*1.05, 100)
ax4.plot(x_fit, slope*x_fit + intercept, '--', color='red', 
         label=f'Linear Fit (R²={r_value**2:.3f})')
ax4.scatter(0, intercept, color='green', s=120, marker='*', zorder=6,
            label=f'Continuum ≈ {intercept:.1f}')
ax4.set_xlabel('1/M')
ax4.set_ylabel('Mean G')
ax4.set_title('Extrapolation of Mean G')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("HSMT_Finite_Size_Scaling_Summary.png", dpi=150, bbox_inches='tight')
print("\nSummary figure saved as: HSMT_Finite_Size_Scaling_Summary.png")

print("\nFinite-size scaling summary complete.")