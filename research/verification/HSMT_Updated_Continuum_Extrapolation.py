#!/usr/bin/env python3
"""
HSMT Phase 4 – Updated Continuum Extrapolation (8 Data Points)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Data (including new M=3000 point)
# ============================================================
M_values   = np.array([300, 600, 900, 1200, 1500, 1800, 2400, 3000])
delta_rho  = np.array([0.02000, 0.01000, 0.00667, 0.00500, 0.00400, 0.00333, 0.00250, 0.00200])
G_values   = np.array([1.5372, 1.0331, 0.8884, 0.7757, 0.6618, 0.5674, 0.5109, 0.4672])
one_over_G = np.array([0.6505, 0.9680, 1.1256, 1.2891, 1.5111, 1.7625, 1.9574, 2.1402])

one_over_M = 1.0 / M_values

print("=" * 95)
print("HSMT Phase 4 – Updated Continuum Extrapolation (8 Data Points)")
print("Fixed physical volume: L ≈ 12")
print("=" * 95)

# ============================================================
# Linear and Quadratic Fits vs Δρ
# ============================================================
p_G_lin   = np.polyfit(delta_rho, G_values,   1)
p_inv_lin = np.polyfit(delta_rho, one_over_G, 1)
p_G_quad   = np.polyfit(delta_rho, G_values,   2)
p_inv_quad = np.polyfit(delta_rho, one_over_G, 2)

G_cont_lin   = np.polyval(p_G_lin,   0.0)
inv_cont_lin = np.polyval(p_inv_lin, 0.0)
G_cont_quad   = np.polyval(p_G_quad,   0.0)
inv_cont_quad = np.polyval(p_inv_quad, 0.0)

# ============================================================
# Linear and Quadratic Fits vs 1/M
# ============================================================
p_G_Mlin   = np.polyfit(one_over_M, G_values,   1)
p_inv_Mlin = np.polyfit(one_over_M, one_over_G, 1)
p_G_Mquad   = np.polyfit(one_over_M, G_values,   2)
p_inv_Mquad = np.polyfit(one_over_M, one_over_G, 2)

G_cont_Mlin   = np.polyval(p_G_Mlin,   0.0)
inv_cont_Mlin = np.polyval(p_inv_Mlin, 0.0)
G_cont_Mquad   = np.polyval(p_G_Mquad,   0.0)
inv_cont_Mquad = np.polyval(p_inv_Mquad, 0.0)

# ============================================================
# Summary Table
# ============================================================
print("\n" + "=" * 95)
print("Estimated Continuum Limit (Δρ → 0 or M → ∞) – Updated with M=3000")
print("=" * 95)
print(f"{'Fit Type':<25} | {'Variable':<10} | {'G (Δρ=0)':>12} | {'1/G (Δρ=0)':>13}")
print("-" * 95)
print(f"{'Linear':<25} | {'Δρ':<10} | {G_cont_lin:>12.4f} | {inv_cont_lin:>13.4f}")
print(f"{'Quadratic':<25} | {'Δρ':<10} | {G_cont_quad:>12.4f} | {inv_cont_quad:>13.4f}")
print(f"{'Linear':<25} | {'1/M':<10} | {G_cont_Mlin:>12.4f} | {inv_cont_Mlin:>13.4f}")
print(f"{'Quadratic':<25} | {'1/M':<10} | {G_cont_Mquad:>12.4f} | {inv_cont_Mquad:>13.4f}")
print("=" * 95)

# ============================================================
# Fit Quality (R²)
# ============================================================
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

G_pred_lin   = np.polyval(p_G_lin,   delta_rho)
G_pred_quad  = np.polyval(p_G_quad,  delta_rho)
inv_pred_lin = np.polyval(p_inv_lin, delta_rho)
inv_pred_quad= np.polyval(p_inv_quad,delta_rho)

print("\n[Fit Quality – R² with 8 points]")
print(f"  G vs Δρ (linear)   : {r_squared(G_values, G_pred_lin):.4f}")
print(f"  G vs Δρ (quadratic): {r_squared(G_values, G_pred_quad):.4f}")
print(f"  1/G vs Δρ (linear) : {r_squared(one_over_G, inv_pred_lin):.4f}")
print(f"  1/G vs Δρ (quadratic): {r_squared(one_over_G, inv_pred_quad):.4f}")
print("=" * 95)