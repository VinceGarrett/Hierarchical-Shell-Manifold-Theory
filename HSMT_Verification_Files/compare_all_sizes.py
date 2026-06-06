import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Load Data from All Four Sizes
# ============================================================
files = {
    800:  "G_results_M800_n1500.npy",
    1000: "G_results_M1000_n1500.npy",
    1200: "G_results_M1200_n1500.npy",
    1500: "G_results_M1500_n1500.npy"
}

data = {}
for m, fname in files.items():
    g = np.load(fname, allow_pickle=True)
    data[m] = g[~np.isnan(g)]
    print(f"M={m}: {len(data[m])} valid samples")

print()

# ============================================================
# Comparative Statistics Table
# ============================================================
def get_stats(d):
    return {
        'mean': np.mean(d),
        'median': np.median(d),
        'std': np.std(d),
        'skewness': stats.skew(d),
        'kurtosis': stats.kurtosis(d),
        'p90': np.percentile(d, 90),
        'p95': np.percentile(d, 95),
        'p99': np.percentile(d, 99),
    }

print("=== Finite-Size Comparison (All High-Statistics Runs) ===")
print(f"{'Metric':<12}", end="")
for m in [800, 1000, 1200, 1500]:
    print(f"M={m:>6}", end="   ")
print()
print("-" * 70)

metrics = ['mean', 'median', 'std', 'skewness', 'kurtosis', 'p90', 'p95', 'p99']
for metric in metrics:
    print(f"{metric:<12}", end="")
    for m in [800, 1000, 1200, 1500]:
        val = get_stats(data[m])[metric]
        print(f"{val:>8.2f}", end="   ")
    print()

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
colors = {800: 'blue', 1000: 'red', 1200: 'green', 1500: 'purple'}

# Overlaid Histograms
ax1 = axes[0, 0]
for m in [800, 1000, 1200, 1500]:
    ax1.hist(data[m], bins=60, density=True, alpha=0.35, color=colors[m], label=f'M={m}')
ax1.set_xlabel('Geometric Criterion G')
ax1.set_ylabel('Density')
ax1.set_title('Distribution Comparison (M=800 → 1500)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box Plot
ax2 = axes[0, 1]
bp = ax2.boxplot([data[800], data[1000], data[1200], data[1500]],
                 tick_labels=['M=800', 'M=1000', 'M=1200', 'M=1500'],
                 patch_artist=True)
for patch, color in zip(bp['boxes'], [colors[800], colors[1000], colors[1200], colors[1500]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax2.set_ylabel('G')
ax2.set_title('Box Plot Comparison')
ax2.grid(True, alpha=0.3)

# Tail Comparison P(G > t)
ax3 = axes[1, 0]
thresholds = np.linspace(50, 600, 120)
for m in [800, 1000, 1200, 1500]:
    tail = [np.sum(data[m] > t) / len(data[m]) for t in thresholds]
    ax3.plot(thresholds, tail, label=f'M={m}', color=colors[m])
ax3.set_xlabel('G Threshold')
ax3.set_ylabel('Fraction of Samples > Threshold')
ax3.set_title('Tail Comparison P(G > t)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Mean & Median vs System Size
ax4 = axes[1, 1]
ms = [800, 1000, 1200, 1500]
means = [np.mean(data[m]) for m in ms]
medians = [np.median(data[m]) for m in ms]
ax4.plot(ms, means, 'o-', color='blue', markersize=8, label='Mean G')
ax4.plot(ms, medians, 's--', color='red', markersize=8, label='Median G')
ax4.set_xlabel('Lattice Size M')
ax4.set_ylabel('G')
ax4.set_title('Mean and Median G vs System Size')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("G_finite_size_comparison_all.png", dpi=150, bbox_inches='tight')
print("\nComparison plot saved as: G_finite_size_comparison_all.png")

print("\nFinite-size comparison complete.")