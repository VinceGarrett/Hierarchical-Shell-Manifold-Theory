"""A1 declaration contract.

A residual is not computed until the convention, the conventional sector,
and the subtraction order have been named.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Declaration:
    convention: str          # "resolvent" or "driven"
    subtraction_order: int   # fixed before the curve is inspected; default <= 1
    known_loss: str          # "absent" or a named independent measurement
    parasitic_pole_bound: int
    tuning_map: str          # name of T_j, or "single setting"

    def __post_init__(self):
        if self.convention not in {"resolvent", "driven"}:
            raise ValueError("convention must be 'resolvent' or 'driven'")
        if self.subtraction_order < 0 or self.subtraction_order > 1:
            raise ValueError("minimal class allows n <= 1; raising n is a new declaration")
        if self.parasitic_pole_bound < 0:
            raise ValueError("pole bound must be nonnegative")
        if not self.known_loss or not self.tuning_map:
            raise ValueError("known loss and tuning map must be named, even if absent")


def residual(sigma_rec: np.ndarray, sigma_c: np.ndarray, decl: Declaration) -> np.ndarray:
    """Sigma_R = Sigma_rec - Sigma_C^declared. decl is required and unused only as a gate."""
    if not isinstance(decl, Declaration):
        raise TypeError("a Declaration is required before a residual is formed")
    return np.asarray(sigma_rec) - np.asarray(sigma_c)


def density(sigma_r: np.ndarray, decl: Declaration) -> np.ndarray:
    if decl.convention not in {"resolvent", "driven"}:
        raise ValueError("convention missing")
    return -np.imag(sigma_r) / np.pi


if __name__ == "__main__":
    decl = Declaration("resolvent", 1, "absent", 0, "single setting")
    z = np.linspace(-1, 1, 5) + 1j * 0.1
    sigma_rec = 0.2 / z
    sigma_c = 0.05 + 0j * z
    r = residual(sigma_rec, sigma_c, decl)
    print("declaration accepted")
    print("residual imag unchanged by real constant:", np.max(np.abs(np.imag(r) - np.imag(sigma_rec))))
    try:
        Declaration("resolvent", 3, "absent", 0, "single setting")
    except ValueError as exc:
        print("raised order rejected:", exc)
