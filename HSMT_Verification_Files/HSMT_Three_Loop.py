import sympy as sp
from sympy import symbols, pi

print("=== HSMT Beta Function up to Three Loops ===\n")

# Symbols
lam = symbols('lambda_eff', positive=True)
d_avg = symbols('d_avg')          # <d(rho) - 4>_w

# Loop factor
L = 1 / (16 * pi**2)

# One-loop term
beta_1L = L * lam**2 * (2 + d_avg)

# Two-loop term
beta_2L = L**2 * lam**3 * (3 + 4 * d_avg)

# Three-loop term (from our calculation)
beta_3L = L**3 * lam**4 * (sp.Rational(16,3) + 4 * d_avg)

# Full beta function up to 3 loops
beta_full = beta_1L + beta_2L + beta_3L

print("Full beta function up to three-loop order:\n")
sp.pprint(beta_full)

print("\n=== Separated by Order ===")
print("\nOne-loop:")
sp.pprint(beta_1L)

print("\nTwo-loop:")
sp.pprint(beta_2L)

print("\nThree-loop (leading):")
sp.pprint(beta_3L)