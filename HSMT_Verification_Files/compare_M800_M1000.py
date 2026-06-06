import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Load Data
# ============================================================
file_800 = "G_results_M800_n1500.npy"
file_1000 = "G_results_M1000_n1500.npy"

g800 = np.load(file_800, allow_pickle=True)
g1000 = np.load(file_1000, allow_pickle=True)

g800_valid = g800[~np.isnan(g800)]
g1000_valid = g1000[~np.isnan(g1000)]

print(f"M=800  : {len(g800_valid)} valid samples")
print(f"M=1000 : {len(g1000_valid)} valid samples\n")

# ============================================================
# Comparative Statistics
# ============================================================
def get_stats(data, label):
    return {
        'label': label,
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'p90': np.percentile(data, 90),
        'p95': np.percentile(data, 95),
        'p99': np.percentile(data, 99),
    }

stats800 = get_stats(g800_valid, "M=800")
stats1000 = get_stats(g1000_valid, "M=1000")

print("=== Comparative Statistics ===")
print(f"{'Metric':<12} {'M=800':>12} {'M=1000':>12} {'Difference':>12}")
print("-" * 50)
for key in ['mean', 'median', 'std', 'skewness', 'kurtosis', 'p90', 'p95', 'p99']:
    diff = stats1000[key] - stats800[key]
    print(f"{key:<12} {stats800[key]:>12.3f} {stats1000[key]:>12.3f} {diff:>12.3f}")

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Overlaid Histograms
ax1 = axes[0, 0]
ax1.hist(g800_valid, bins=60, density=True, alpha=0.5, color='blue', label='M=800')
ax1.hist(g1000_valid, bins=60, density=True, alpha=0.5, color='red', label='M=1000')
ax1.set_xlabel('Geometric Criterion G')
ax1.set_ylabel('Density')
ax1.set_title('Distribution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box Plot
ax2 = axes[0, 1]
ax2.boxplot([g800_valid, g1000_valid], labels=['M=800', 'M=1000'])
ax2.set_ylabel('G')
ax2.set_title('Box Plot Comparison')
ax2.grid(True, alpha=0.3)

# Tail Comparison (Cumulative)
ax3 = axes[1, 0]
thresholds = np.linspace(50, 400, 100)
tail800 = [np.sum(g800_valid > t) / len(g800_valid) for t in thresholds]
tail1000 = [np.sum(g1000_valid > t) / len(g1000_valid) for t in thresholds]
ax3.plot(thresholds, tail800, label='M=800', color='blue')
ax3.plot(thresholds, tail1000, label='M=1000', color='red')
ax3.set_xlabel('G Threshold')
ax3.set_ylabel('Fraction of Samples > Threshold')
ax3.set_title('Tail Comparison (P(G > t))')
ax3.legend()
ax3.grid(True, alpha=0.3)

# QQ-plot style comparison (using Gamma fit from M=800 as reference)
ax4 = axes[1, 1]
gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(g800_valid, floc=0)
gamma_dist = stats.gamma(gamma_shape, loc=gamma_loc, scale=gamma_scale)
stats.probplot(g1000_valid, dist=gamma_dist, plot=ax4)
ax4.set_title('QQ-plot: M=1000 vs Gamma fit from M=800')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("G_comparison_M800_vs_M1000.png", dpi=150, bbox_inches='tight')
print("\nComparison plot saved as: G_comparison_M800_vs_M1000.png")

print("\nComparison analysis complete.")