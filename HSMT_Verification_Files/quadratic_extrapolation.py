import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

# ============================================================
# Data
# ============================================================
Ms = np.array([800, 1000, 1200, 1500])
invM = 1.0 / Ms
invM2 = invM**2

quantities = {
    'Mean G':     np.array([96.03, 98.62, 110.34, 129.51]),
    'Median G':   np.array([70.91, 74.10, 89.85, 101.66]),
    'Skewness':   np.array([2.58, 2.29, 1.91, 1.82]),
    'Kurtosis':   np.array([9.09, 5.60, 4.97, 4.49]),
}

print("=== Quadratic Extrapolation (a + b/M + c/M²) ===\n")

results = {}

for name, y in quantities.items():
    # Fit y = a + b*x + c*x² where x = 1/M
    coeffs = polyfit(invM, y, 2)   # returns [a, b, c]
    a, b, c = coeffs
    results[name] = {'continuum': a, 'b': b, 'c': c}
    
    # Simple quality measure (R²-like)
    y_pred = a + b*invM + c*invM2
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"{name:12} → Continuum ≈ {a:8.2f}   "
          f"(b = {b:8.1f}, c = {c:8.1f}, R² ≈ {r2:.3f})")

print("\nQuadratic extrapolation complete.")