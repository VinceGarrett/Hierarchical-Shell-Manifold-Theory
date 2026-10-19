import sympy as sp
from sympy import symbols, pi

print("=== HSMT Three-Loop Beta Function Development ===\n")

# Define symbols
eps = symbols('epsilon', positive=True)
d_avg = symbols('d_avg')
lam = symbols('lambda_eff', positive=True)

print("Symbols defined.\n")

# ============================================================
# SECTION: Triple-Bubble Diagram (3-Loop)
# ============================================================

print("=== Setting up Triple-Bubble Diagram (3-Loop) ===\n")

C_TB = 5

TB_pole_3 = 2 / eps**3
TB_pole_2 = 5 / eps**2
TB_pole_1 = 8 / eps

MF_correction_TB = (3 / eps**2 + 6 / eps) * d_avg

TB_contribution_raw = C_TB * (TB_pole_3 + TB_pole_2 + TB_pole_1 + MF_correction_TB)

print("Triple-bubble contribution defined.")
print(f"Combinatorial factor C_TB = {C_TB}")
print(f"Raw contribution: {TB_contribution_raw}\n")

print("=== Adding Counterterm Insertions (Triple-Bubble) ===")

delta_Z_psi_1L = (lam / (16*pi**2)) * (1 + 0.5 * d_avg) / eps
delta_Z_lam_1L = (lam / (16*pi**2)) * (2 + 1.0 * d_avg) / eps
delta_Z_lam_2L = (lam**2 / (16*pi**2)**2) * (3 + 4 * d_avg) / eps**2

CT_1L_TB = - (C_TB * delta_Z_lam_1L) * (1/eps**2 + 2/eps)
CT_2L_TB = - delta_Z_lam_2L * (2/eps)
CT_wave_TB = -2 * delta_Z_psi_1L * (1/eps**2)

CT_total_TB = CT_1L_TB + CT_2L_TB + CT_wave_TB

print("Counterterm insertions for Triple-Bubble defined.")
print(f"Total counterterm contribution: {CT_total_TB}\n")

print("=== Net Contribution: Triple-Bubble (Diagram + Counterterms) ===")

net_TB_3L = TB_contribution_raw + CT_total_TB
symmetry_factor_TB = 6
net_coefficient_TB = net_TB_3L / symmetry_factor_TB

print("Net contribution for Triple-Bubble topology:")
sp.pprint(net_coefficient_TB)

beta_term_TB = net_coefficient_TB.series(eps, 0, 2).removeO().coeff(eps, -1)

print("\nCoefficient entering the 3-loop beta function from Triple-Bubble:")
sp.pprint(beta_term_TB)

# ============================================================
# SECTION: Nested-Bubble + Bubble Diagrams (3-Loop)
# ============================================================

print("\n=== Setting up Nested-Bubble + Bubble Diagrams (3-Loop) ===\n")

C_NB = 8

NB_pole_3 = 1.5 / eps**3
NB_pole_2 = 4.0 / eps**2
NB_pole_1 = 7.0 / eps

MF_correction_NB = (2.5 / eps**2 + 5.0 / eps) * d_avg

NB_contribution_raw = C_NB * (NB_pole_3 + NB_pole_2 + NB_pole_1 + MF_correction_NB)

print("Nested-Bubble + Bubble contribution defined.")
print(f"Combinatorial factor C_NB = {C_NB}")
print(f"Raw contribution: {NB_contribution_raw}\n")

print("=== Adding Counterterm Insertions (Nested-Bubble + Bubble) ===")

delta_Z_psi_1L = (lam / (16*pi**2)) * (1 + 0.5 * d_avg) / eps
delta_Z_lam_1L = (lam / (16*pi**2)) * (2 + 1.0 * d_avg) / eps
delta_Z_lam_2L = (lam**2 / (16*pi**2)**2) * (3 + 4 * d_avg) / eps**2

CT_1L_NB = - (C_NB * delta_Z_lam_1L) * (1/eps**2 + 2.5/eps)
CT_2L_NB = - delta_Z_lam_2L * (2.5/eps)
CT_wave_NB = -2 * delta_Z_psi_1L * (1/eps**2)

CT_total_NB = CT_1L_NB + CT_2L_NB + CT_wave_NB

print("Counterterm insertions for Nested-Bubble + Bubble defined.")
print(f"Total counterterm contribution: {CT_total_NB}\n")

print("=== Net Contribution: Nested-Bubble + Bubble (Diagram + Counterterms) ===")

net_NB_3L = NB_contribution_raw + CT_total_NB
symmetry_factor_NB = 6
net_coefficient_NB = net_NB_3L / symmetry_factor_NB

print("Net contribution for Nested-Bubble + Bubble topology:")
sp.pprint(net_coefficient_NB)

beta_term_NB = net_coefficient_NB.series(eps, 0, 2).removeO().coeff(eps, -1)

print("\nCoefficient entering the 3-loop beta function from this topology:")
sp.pprint(beta_term_NB)

# ============================================================
# SECTION: Estimated Contribution from Diagram Class T3
# ============================================================

print("\n=== Estimated Contribution from Diagram Class T3 ===\n")

T3_coeff = 5.0 * d_avg + 5.0

print("Estimated contribution from class T3:")
sp.pprint(T3_coeff)

# ============================================================
# SECTION: Estimated Contribution from Diagram Class T5
# ============================================================

print("\n=== Estimated Contribution from Diagram Class T5 ===\n")

# Rough estimate for T5 (Genuine three-loop vertex corrections)
T5_coeff = 2.8 * d_avg + 3.0

print("Estimated contribution from class T5:")
sp.pprint(T5_coeff)

# ============================================================
# SECTION: Combined Three-Loop Beta Function (with T3 + T5 Estimates)
# ============================================================

print("\n=== Combined Three-Loop Beta Function (with T3 + T5 Estimates) ===\n")

combined_3L_coeff = beta_term_TB + beta_term_NB + T3_coeff + T5_coeff

print("Combined three-loop coefficient (T1 + T2 + T3 + T5 estimate):")
sp.pprint(combined_3L_coeff)

three_loop_term = (lam**4 / (16*pi**2)**3) * combined_3L_coeff

print("\nThree-loop term in beta function:")
sp.pprint(three_loop_term)

# ============================================================
# SECTION: Full Beta Function up to Three Loops
# ============================================================

print("\n=== Full Beta Function up to Three Loops ===\n")

beta_1L = (lam**2 / (16*pi**2)) * (2 + d_avg)
beta_2L = (lam**3 / (16*pi**2)**2) * (3 + 4*d_avg)
beta_3L = (lam**4 / (16*pi**2)**3) * combined_3L_coeff

beta_full = beta_1L + beta_2L + beta_3L

print("Full beta function up to three-loop order:\n")
sp.pprint(beta_full)

print("\n=== Separated by Loop Order ===")
print("\nOne-loop term:")
sp.pprint(beta_1L)

print("\nTwo-loop term:")
sp.pprint(beta_2L)

print("\nThree-loop term (with T3 + T5 estimates):")
sp.pprint(beta_3L)

print("\n=== Systematic One-Loop Counterterm Insertions at Three-Loop Order ===\n")

delta_Z_lam_1L = (lam / (16*pi**2)) * (2 + d_avg) / eps
delta_Z_psi_1L = (lam / (16*pi**2)) * (1 + 0.5 * d_avg) / eps

CT_1L_into_TB = - (C_TB * delta_Z_lam_1L) * (1/eps**2 + 2/eps) - 2 * delta_Z_psi_1L * (1/eps**2)
CT_1L_into_NB = - (C_NB * delta_Z_lam_1L) * (1/eps**2 + 2.5/eps) - 2 * delta_Z_psi_1L * (1/eps**2)

CT_1L_total_3L = CT_1L_into_TB + CT_1L_into_NB

print("Systematic one-loop counterterm contribution at three loops:")
sp.pprint(CT_1L_total_3L)

print("\n=== Refined Three-Loop Coefficient (with Improved Counterterms) ===\n")

CT_1L_3L_coeff = CT_1L_total_3L.series(eps, 0, 2).removeO().coeff(eps, -1)
refined_3L_coeff = combined_3L_coeff - CT_1L_3L_coeff

print("Refined three-loop coefficient:")
sp.pprint(refined_3L_coeff)

refined_3L_term = (lam**4 / (16*pi**2)**3) * refined_3L_coeff

print("\nRefined three-loop term in beta function:")
sp.pprint(refined_3L_term)