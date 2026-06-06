import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# ANALYSIS SCRIPT FOR G RESULTS
# ============================================================
# This script reads the file G_results_M800_n1500.npy
# and performs deeper statistical analysis.
#
# Make sure G_results_M800_n1500.npy is in the same folder.
# ============================================================

filename = "G_results_M800_n1500.npy"

# Load data
g_values = np.load(filename, allow_pickle=True)
g_valid = g_values[~np.isnan(g_values)]

print(f"Loaded {len(g_valid)} valid samples from {filename}\n")

# ============================================================
# Advanced Statistics
# ============================================================
def get_advanced_stats(data):
    return {
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'p10': np.percentile(data, 10),
        'p25': np.percentile(data, 25),
        'p75': np.percentile(data, 75),
        'p90': np.percentile(data, 90),
        'p95': np.percentile(data, 95),
        'p99': np.percentile(data, 99),
    }

stats_dict = get_advanced_stats(g_valid)

print("=== Advanced Statistical Summary ===")
for key, value in stats_dict.items():
    print(f"{key:12}: {value:.4f}")

# ============================================================
# Distribution Fitting
# ============================================================
# Fit Gamma distribution
gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(g_valid, floc=0)
gamma_dist = stats.gamma(gamma_shape, loc=gamma_loc, scale=gamma_scale)

# Fit Lognormal distribution
lognorm_shape, lognorm_loc, lognorm_scale = stats.lognorm.fit(g_valid, floc=0)
lognorm_dist = stats.lognorm(lognorm_shape, loc=lognorm_loc, scale=lognorm_scale)

print(f"\nGamma fit     → shape = {gamma_shape:.3f}, scale = {gamma_scale:.3f}")
print(f"Lognormal fit → shape = {lognorm_shape:.3f}, scale = {lognorm_scale:.3f}")

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with fitted distributions
ax1 = axes[0]
ax1.hist(g_valid, bins=60, density=True, alpha=0.6, color='steelblue', 
         edgecolor='black', label='Data')
x = np.linspace(g_valid.min(), g_valid.max() * 1.1, 500)
ax1.plot(x, gamma_dist.pdf(x), 'r-', lw=2, label=f'Gamma')
ax1.plot(x, lognorm_dist.pdf(x), 'g--', lw=2, label=f'Lognormal')
ax1.set_xlabel('Geometric Criterion G')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of G (M=800)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# QQ-plot
ax2 = axes[1]
stats.probplot(g_valid, dist=gamma_dist, plot=ax2)
ax2.set_title('QQ-plot vs Fitted Gamma Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("G_distribution_analysis_M800.png", dpi=150, bbox_inches='tight')
print("\nPlot saved as: G_distribution_analysis_M800.png")

# ============================================================
# Tail Analysis
# ============================================================
print("\n=== Tail Analysis ===")
for threshold in [150, 200, 300, 400]:
    fraction = np.sum(g_valid > threshold) / len(g_valid)
    print(f"Fraction of samples with G > {threshold:3d}: {fraction*100:5.2f}%")

print("\nAnalysis complete.")