import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.stats import linregress

# ============================================================
# Data from All Five Sizes
# ============================================================
Ms = np.array([800, 1000, 1200, 1500, 2000])
invM = 1.0 / Ms
invM2 = invM**2

# Quantities (Mean G, Median G, Skewness, Kurtosis)
quantities = {
    'Mean G':   np.array([96.03, 98.62, 110.34, 129.51, 150.39]),
    'Median G': np.array([70.91, 74.10, 89.85, 101.66, 121.38]),
    'Skewness': np.array([2.58, 2.29, 1.91, 1.82, 1.65]),
    'Kurtosis': np.array([9.09, 5.60, 4.97, 4.49, 4.12]),
}

print("=== Finite-Size Extrapolation (5 data points) ===\n")

for name, y in quantities.items():
    print(f"--- {name} ---")
    
    # Linear fit (a + b/M)
    slope, intercept, r_value, _, _ = linregress(invM, y)
    lin_continuum = intercept
    print(f"Linear (1/M)     → Continuum ≈ {lin_continuum:8.2f}   (R² = {r_value**2:.3f})")
    
    # Quadratic fit (a + b/M + c/M²)
    coeffs = polyfit(invM, y, 2)   # [a, b, c]
    a, b, c = coeffs
    y_pred = a + b*invM + c*invM2
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    quad_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    print(f"Quadratic        → Continuum ≈ {a:8.2f}   (R² = {quad_r2:.3f})")
    
    print()

print("Extrapolation complete.")