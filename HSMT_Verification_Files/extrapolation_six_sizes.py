import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.stats import linregress

# ============================================================
# Data from All Six Sizes
# ============================================================
Ms = np.array([800, 1000, 1200, 1500, 2000, 2500])
invM = 1.0 / Ms
invM2 = invM**2

quantities = {
    'Mean G':   np.array([96.03, 98.62, 110.34, 129.51, 150.39, 150.95]),
    'Median G': np.array([70.91, 74.10, 89.85, 101.66, 121.38, 109.24]),
    'Skewness': np.array([2.58, 2.29, 1.91, 1.82, 1.65, 1.72]),
    'Kurtosis': np.array([9.09, 5.60, 4.97, 4.49, 4.12, 5.85]),
}

print("=== Finite-Size Extrapolation (6 data points) ===\n")

for name, y in quantities.items():
    print(f"--- {name} ---")
    
    # Linear fit
    slope, intercept, r_value, _, _ = linregress(invM, y)
    print(f"Linear (1/M) → Continuum ≈ {intercept:8.2f}   (R² = {r_value**2:.3f})")
    
    # Quadratic fit
    coeffs = polyfit(invM, y, 2)
    a, b, c = coeffs
    y_pred = a + b*invM + c*invM2
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    print(f"Quadratic    → Continuum ≈ {a:8.2f}   (R² = {r2:.3f})")
    
    print()

print("Extrapolation with 6 points complete.")