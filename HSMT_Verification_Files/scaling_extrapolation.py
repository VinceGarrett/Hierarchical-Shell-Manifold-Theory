import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ============================================================
# Data from High-Statistics Runs
# ============================================================
Ms = np.array([800, 1000, 1200, 1500])
invM = 1.0 / Ms

# Key quantities (from the comparison runs)
mean_G     = np.array([96.03, 98.62, 110.34, 129.51])
median_G   = np.array([70.91, 74.10, 89.85, 101.66])
skewness   = np.array([2.58, 2.29, 1.91, 1.82])
kurtosis   = np.array([9.09, 5.60, 4.97, 4.49])
p90        = np.array([179.40, 195.19, 197.96, 239.63])
p95        = np.array([236.28, 257.05, 231.79, 291.15])
p99        = np.array([371.09, 357.00, 346.18, 396.68])

quantities = {
    'Mean G': mean_G,
    'Median G': median_G,
    'Skewness': skewness,
    'Kurtosis': kurtosis,
    'p90': p90,
    'p95': p95,
    'p99': p99
}

print("=== Linear Extrapolation vs 1/M (M → ∞) ===\n")

results = {}

for name, y in quantities.items():
    slope, intercept, r_value, p_value, std_err = linregress(invM, y)
    continuum_value = intercept          # value at 1/M = 0
    results[name] = {
        'continuum': continuum_value,
        'slope': slope,
        'R2': r_value**2
    }
    print(f"{name:12} → Continuum ≈ {continuum_value:8.2f}   "
          f"(slope = {slope:7.3f}, R² = {r_value**2:.3f})")

# ============================================================
# Optional Plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

plot_keys = ['Mean G', 'Median G', 'Kurtosis', 'p99']
for idx, key in enumerate(plot_keys):
    ax = axes[idx]
    y = quantities[key]
    slope, intercept, r_value, _, _ = linregress(invM, y)
    
    ax.scatter(invM, y, color='black', s=60, zorder=5, label='Data')
    
    # Fit line
    x_fit = np.linspace(0, invM.max() * 1.05, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, '--', color='red', label=f'Fit (R²={r_value**2:.3f})')
    
    # Extrapolated point
    ax.scatter(0, intercept, color='green', s=100, marker='*', zorder=6,
               label=f'Continuum ≈ {intercept:.1f}')
    
    ax.set_xlabel('1/M')
    ax.set_ylabel(key)
    ax.set_title(f'{key} vs 1/M')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("G_scaling_extrapolation.png", dpi=150, bbox_inches='tight')
print("\nExtrapolation plot saved as: G_scaling_extrapolation.png")

print("\nSimple linear extrapolation complete.")