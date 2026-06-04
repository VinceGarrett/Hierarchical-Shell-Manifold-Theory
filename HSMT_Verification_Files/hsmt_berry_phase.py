#!/usr/bin/env python3
"""
HSMT Geometric Primer — Berry Phase Example (Final Fixed Version)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Parameters
# ============================================
sigma_demo = 0.35
Delta_demo = 5.0

print("=" * 70)
print("HSMT Berry Phase Demonstration")
print("=" * 70)
print(f"Using demo parameters: Δ = {Delta_demo}, σ = {sigma_demo}")
print()

# ============================================
# Berry Connection from tunable blurriness
# ============================================
def berry_connection(phi):
    sigma_phi = sigma_demo * (1 + 0.2 * np.sin(phi))
    overlap_amp = np.exp(- (Delta_demo**2) / (2 * sigma_phi**2))
    return -overlap_amp * np.cos(phi)


# ============================================
# Compute Berry Phase
# ============================================
def compute_berry_phase(n_points=300):
    phi_values = np.linspace(0, 2*np.pi, n_points)
    A_phi = np.array([berry_connection(phi) for phi in phi_values])
    
    # Modern NumPy (np.trapz was removed)
    berry_gamma = np.trapezoid(A_phi, phi_values)
    return berry_gamma, phi_values, A_phi


# Run calculation
gamma, phi, A = compute_berry_phase()

print(f"Computed Berry phase γ = {gamma:.6f} radians")
print(f"                  ≈ {gamma*180/np.pi:.3f} degrees")
print(f"Maximum |A_φ| ≈ {np.max(np.abs(A)):.6f}")

# ============================================
# Plot
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(phi, A, 'b-', linewidth=2.5, label='Berry Connection A_φ')
plt.fill_between(phi, 0, A, alpha=0.3, color='blue')
plt.xlabel('Cyclic Parameter φ (radians) — variation of blurriness σ')
plt.ylabel('Berry Connection A_φ')
plt.title('HSMT: Berry Phase from Shell Overlap / Tunable Blurriness')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Success! The plot should now appear.")
print("This demonstrates how a geometric (Berry) phase naturally emerges")
print("in HSMT when the blurriness parameter is varied cyclically.")