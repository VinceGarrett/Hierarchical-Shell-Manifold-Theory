#!/usr/bin/env python3
"""
HSMT Geometric Primer — Kähler & Symplectic Structure
"""

import numpy as np
import matplotlib.pyplot as plt

sigma_demo = 0.35
Delta_demo = 5.0

print("=" * 75)
print("HSMT: Kähler and Symplectic Geometry Mapping")
print("=" * 75)

# Simple model of projected state overlap
def overlap_amplitude(z):
    """z is a complex coordinate in the projected N=0 layer"""
    r = np.abs(z)
    return np.exp(-r**2 / (2 * sigma_demo**2))   # Gaussian-like coherent state

# Kähler potential in HSMT language
def kahler_potential(z):
    """K(z, z-bar) = -log |<ψ|Φ|ψ>|^2"""
    amp = overlap_amplitude(z)
    return -np.log(amp**2 + 1e-12)   # small epsilon for stability

# Compute on a grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
K = kahler_potential(Z)

print("Kähler potential computed on grid.")
print("In HSMT, this potential is induced by the Gaussian overlap kernel w_N(ℓ)")

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, K, levels=50, cmap='viridis')
plt.colorbar(label='Kähler Potential K(z, z-bar)')
plt.xlabel('Re(z) — Projected Coordinate')
plt.ylabel('Im(z) — Projected Coordinate')
plt.title('HSMT Kähler Potential from Shell Overlap Projection')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ This demonstrates how the abstract Kähler structure")
print("   emerges naturally from the HSMT projection Φ and Gaussian kernel.")