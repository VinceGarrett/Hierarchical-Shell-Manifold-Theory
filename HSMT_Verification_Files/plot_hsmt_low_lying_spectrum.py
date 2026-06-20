import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
import mpmath as mp

# Set high precision for hypergeometric evaluation
mp.mp.dps = 25

# ==============================================================================
# Parameters (from HSMT Verification Suite v10.0)
# ==============================================================================
alpha = mp.mpf('0.180000')
sigma0 = mp.mpf('0.3535533905932738')   # sqrt(2)/4

# Ground state (n=0) analytic parameters
beta0 = mp.mpf('0.2494')
poly_coeffs = [mp.mpf('0.0'), mp.mpf('0.0014'), mp.mpf('-0.0001'), mp.mpf('0.0011')]

# n=1 hypergeometric fit parameters (from hybrid reconstruction)
b1 = mp.mpf('4.264102')
c1 = mp.mpf('4.429937')
kappa1 = mp.mpf('4.0') + (4 * mp.pi / 3)   # approximate for n=1

# ==============================================================================
# Wavefunction definitions
# ==============================================================================
def psi_n0_analytic(rho):
    """Analytic ground state wavefunction (n=0)"""
    rho_m = mp.mpf(rho)
    g = mp.exp(-beta0 * rho_m**2 / 2)
    p = (poly_coeffs[0] +
         poly_coeffs[1] * rho_m**2 +
         poly_coeffs[2] * rho_m**4 +
         poly_coeffs[3] * rho_m**6)
    return float(g * p)

def hypergeometric_model(rho, n, b, c, kappa):
    """Hypergeometric reconstruction used in hybrid method"""
    rho_m = mp.mpf(rho)
    pref = mp.exp(-alpha * rho_m / 2) * (1 + mp.exp(2 * alpha * rho_m)) ** (-kappa)
    hyp = mp.hyp2f1(-n, b, c, -mp.exp(2 * alpha * rho_m))
    return float(pref * hyp)

def recenter_wavefunction(rho_vals, psi, strength=3.85):
    """Apply the same recentering used in the verification suite"""
    prob = psi**2
    norm = np.trapezoid(prob, rho_vals)
    if norm < 1e-30:
        return psi
    center = np.trapezoid(rho_vals * prob, rho_vals) / norm
    shift = -center
    tilt = np.exp(strength * rho_vals)
    psi_new = psi * tilt
    psi_new /= np.sqrt(np.trapezoid(psi_new**2, rho_vals))
    return psi_new

# ==============================================================================
# Generate data
# ==============================================================================
rho = np.linspace(-12, 12, 2000)

# Ground state (n=0)
psi0 = np.array([psi_n0_analytic(r) for r in rho])
prob0 = psi0**2 / np.trapezoid(psi0**2, rho)

# Excited state (n=1) - reconstructed + recentered
psi1_raw = np.array([hypergeometric_model(r, 1, b1, c1, kappa1) for r in rho])
psi1 = recenter_wavefunction(rho, psi1_raw, strength=3.85)
prob1 = psi1**2 / np.trapezoid(psi1**2, rho)

# ==============================================================================
# Plotting
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(rho, prob0, label=r'$n=0$ (analytic ground state)', linewidth=2.2, color='#1f77b4')
ax.plot(rho, prob1, label=r'$n=1$ (hybrid + recentered)', linewidth=2.2, color='#d62728', linestyle='--')

ax.set_xlabel(r'$\rho$', fontsize=13)
ax.set_ylabel(r'$|\psi(\rho)|^2$', fontsize=13)
ax.set_title('HSMT Low-Lying Spectrum: Probability Densities', fontsize=14, pad=12)
ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
ax.set_xlim(-10, 10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hsmt_low_lying_spectrum_n0_n1.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved as: hsmt_low_lying_spectrum_n0_n1.pdf")