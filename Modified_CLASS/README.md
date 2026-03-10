# Modified CLASS Boltzmann Code

This folder contains the modifications to the CLASS Boltzmann code used for all cosmological predictions in the HSMT paper.

Key changes:
- Projection kernel κ(ℓ) added to the photon Boltzmann hierarchy
- Leakage source terms for inter-shell coupling
- Running gravitational coupling G_N(ℓ) and effective cosmological term Λ_N(ℓ)
- Partition-blending module for BBN (resolves lithium problem)

The code reproduces the global MCMC fits (H₀ = 70.2 ± 1.1 km s⁻¹ Mpc⁻¹) and all luminosity-distance relations without metric expansion.

Instructions:
- Copy the modified files into your CLASS installation
- Use the provided parameters.ini
- Run with `class hsmt.ini`

Full modified source will be added shortly (CLASS is large). The current files show the exact changes described in the paper.
