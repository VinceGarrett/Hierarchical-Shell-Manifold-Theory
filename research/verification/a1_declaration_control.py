"""Verified A1 declaration-control calculator.

Purpose:
Show that positivity and Hilbert consistency of a residual do not identify
the conventional subtraction. Uses the frozen C1 convention only:
    G(z) = 1 / (z - omega0 - Sigma(z))
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import hilbert

ETA = 0.02
OMEGA0 = 0.3
G2 = 0.4
REAL_CONSTANT = 0.25
LOSS_GAMMA = 0.12

def stieltjes(zvals, grid, density, weight):
    out = np.empty_like(zvals, dtype=complex)
    for i, z in enumerate(zvals):
        out[i] = weight * trapezoid(density / (z - grid), grid)
    return out

def density_from_sigma(sigma):
    return -np.imag(sigma) / np.pi

def main():
    # Target positive finite measure: normalized semicircle on [-1,1].
    M = np.linspace(-1.0, 1.0, 5001)
    rho = (2 / np.pi) * np.sqrt(np.clip(1 - M**2, 0, None))

    E = np.linspace(-1.5, 1.5, 6001)
    z = E + 1j * ETA
    sigma_bath = stieltjes(z, M, rho, G2)

    # Conventional pieces.
    constant = REAL_CONSTANT
    loss = -1j * LOSS_GAMMA * np.ones_like(z)
    sigma_rec = constant + loss + sigma_bath

    # Exact C1 inversion.
    G = 1 / (z - OMEGA0 - sigma_rec)
    sigma_inv = z - OMEGA0 - 1 / G
    inversion_error = np.max(np.abs(sigma_inv - sigma_rec))

    # Correct declaration.
    sigma_r_correct = sigma_rec - (constant + loss)

    # Deliberately miss only the real constant.
    sigma_r_miss_constant = sigma_rec - loss
    rho_correct = density_from_sigma(sigma_r_correct)
    rho_miss_constant = density_from_sigma(sigma_r_miss_constant)

    # Deliberately miss only the dissipative term.
    sigma_r_miss_loss = sigma_rec - constant
    rho_miss_loss = density_from_sigma(sigma_r_miss_loss)

    # Stronger counterexample: an undeclared ordinary positive conventional bath.
    Mc = np.linspace(-3.0, 3.0, 6001)
    rho_c = np.exp(-(Mc / 1.2) ** 2)
    rho_c /= trapezoid(rho_c, Mc)
    sigma_conv_bath = stieltjes(z, Mc, rho_c, 0.15)
    rho_with_undeclared_bath = density_from_sigma(sigma_bath + sigma_conv_bath)

    # Finite-window Hilbert check on a wider grid.
    Ew = np.linspace(-8.0, 8.0, 32769)
    zw = Ew + 1j * ETA
    target_w = stieltjes(zw, M, rho, G2)
    conv_w = stieltjes(zw, Mc, rho_c, 0.15)
    mask = np.abs(Ew) < 1.2

    def hilbert_rmse(sigma):
        H_im = np.imag(hilbert(np.imag(sigma)))
        pred_re = -H_im
        residual = np.real(sigma[mask]) - pred_re[mask]
        p0 = residual.mean()
        rmse = np.sqrt(np.mean((residual - p0) ** 2))
        return p0, rmse

    p0_target, rmse_target = hilbert_rmse(target_w)
    p0_wrong, rmse_wrong = hilbert_rmse(target_w + conv_w)
    p0_const, rmse_const = hilbert_rmse(target_w + REAL_CONSTANT)

    print(f"C1 inversion max abs error: {inversion_error:.3e}")
    print(f"Missing real constant: max density change = {np.max(np.abs(rho_miss_constant-rho_correct)):.3e}")
    print(f"Missing real constant: Re shift = {np.median(np.real(sigma_r_miss_constant-sigma_r_correct)):.6f}")
    print(f"Missing loss: density increment = {np.mean(rho_miss_loss-rho_correct):.9f}")
    print(f"Expected gamma/pi = {LOSS_GAMMA/np.pi:.9f}")
    print(f"Minimum density, correctly declared = {rho_correct.min():.9f}")
    print(f"Minimum density, missed loss = {rho_miss_loss.min():.9f}")
    print(f"Minimum density, undeclared positive conventional bath = {rho_with_undeclared_bath.min():.9f}")
    print(f"Hilbert RMSE, target bath = {rmse_target:.6e}, best P0 = {p0_target:.6f}")
    print(f"Hilbert RMSE, target + undeclared conventional bath = {rmse_wrong:.6e}, best P0 = {p0_wrong:.6f}")
    print(f"Hilbert RMSE, target + real constant = {rmse_const:.6e}, best P0 = {p0_const:.6f}")

    # Contract assertions.
    assert inversion_error < 1e-12
    assert np.max(np.abs(rho_miss_constant-rho_correct)) < 1e-14
    assert abs(np.median(np.real(sigma_r_miss_constant-sigma_r_correct)) - REAL_CONSTANT) < 1e-12
    assert np.all(rho_miss_loss > 0)
    assert np.all(rho_with_undeclared_bath > 0)
    assert abs(p0_const - REAL_CONSTANT) < 5e-3

    print("A1 declaration-control assertions: PASS")

if __name__ == "__main__":
    main()
