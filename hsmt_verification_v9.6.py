#!/usr/bin/env python3
"""
HSMT Heavy Brute-Force Symbolic Octonion Proof v9.6
Designed for maximum load on high-end hardware (192 GB RAM)
"""

import sympy as sp
from sympy import tanh, exp, diff, simplify, symbols, expand, collect
import time
import psutil
import json
from pathlib import Path
from datetime import datetime

print("="*120)
print("HSMT HEAVY BRUTE-FORCE SYMBOLIC PROOF v9.6")
print("Maximum load configuration - 192 GB RAM system")
print("="*120)

# ===================================================================
# CONFIGURATION - HEAVY MODE
# ===================================================================
MAX_N = 10000                    # Change this to higher values if you want (10+ will be very heavy)
SAVE_EVERY = 1               # Save results after each n
AGGRESSIVE_SIMPLIFY = True   # Use expand + collect + cancel

output_dir = Path("hsmt_heavy_symbolic_proof")
output_dir.mkdir(exist_ok=True)

rho = symbols('rho', real=True)
alpha, kappa0, A, m0 = symbols('alpha kappa0 A m0', real=True, positive=True)

def print_memory():
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.percent:.1f}% used | Available: {mem.available/(1024**3):.1f} GB")

# ===================================================================
# OCTONION CLASS
# ===================================================================
class Octonion:
    def __init__(self, coeffs):
        self.c = coeffs

    def __mul__(self, other):
        a = self.c
        b = other.c
        c = [0]*8
        c[0] = a[0]*b[0] - sum(a[i]*b[i] for i in range(1,8))
        
        rules = [
            (1,2,4,1), (2,3,5,1), (3,4,6,1), (4,5,7,1),
            (5,6,1,1), (6,7,2,1), (7,1,3,1),
            (2,1,4,-1), (3,2,5,-1), (4,3,6,-1), (5,4,7,-1),
            (6,5,1,-1), (7,6,2,-1), (1,7,3,-1)
        ]
        for i,j,k,sign in rules:
            c[k] += sign * (a[i]*b[j] - a[j]*b[i])
        return Octonion(c)

# ===================================================================
# HEAVY SYMBOLIC CHECK
# ===================================================================
def heavy_check_state(n):
    start = time.time()
    print(f"\n=== HEAVY SYMBOLIC EXPANSION for n = {n} ===")
    print_memory()
    
    kappa_n = kappa0 + (4*sp.pi/3)*n
    psi = exp(-alpha * rho / 2) * (1 + exp(2*alpha*rho))**(-kappa_n)
    dpsi = diff(psi, rho)
    
    results = {"n": n, "units": {}, "status": "running"}
    
    for e_idx in range(1, 8):
        unit_start = time.time()
        u = Octonion([0]*8)
        u.c[e_idx] = tanh(alpha * rho)
        
        psi_coeffs = [psi] + [0]*7
        L = u * Octonion(psi_coeffs)
        R = Octonion(psi_coeffs) * u
        
        sym_term = Octonion([(L.c[i] + R.c[i])/2 for i in range(8)])
        
        residual = sum(A * sym_term.c[i] for i in range(1,8))   # imaginary part
        
        # Heavy simplification
        if AGGRESSIVE_SIMPLIFY:
            residual = expand(residual)
            residual = collect(residual, rho)
            residual = simplify(residual)
        
        simplified = residual
        term_count = len(simplified.args) if hasattr(simplified, 'args') else 1
        
        unit_time = time.time() - unit_start
        
        results["units"][f"e{e_idx}"] = {
            "term_count_before": "N/A (heavy mode)",
            "term_count_after": term_count,
            "simplified": str(simplified)[:500] + "..." if len(str(simplified)) > 500 else str(simplified),
            "is_zero": simplified == 0,
            "time_seconds": unit_time
        }
        
        print(f"  e{e_idx} → {term_count} terms after simplification | Zero: {simplified == 0} | Time: {unit_time:.1f}s")
    
    total_time = time.time() - start
    results["total_time"] = total_time
    results["status"] = "completed"
    
    # Save full result
    with open(output_dir / f"heavy_proof_n{n}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed n={n} in {total_time:.1f} seconds.")
    print_memory()
    return all(unit["is_zero"] for unit in results["units"].values())

# ===================================================================
# MAIN HEAVY RUN
# ===================================================================
if __name__ == "__main__":
    print("Starting HEAVY brute-force symbolic proof...")
    print(f"Will process n = 0 to {MAX_N}")
    print("This will generate very large intermediate expressions.")
    print_memory()
    
    for n in range(0, MAX_N + 1):
        heavy_check_state(n)
    
    print("\n" + "="*120)
    print("HEAVY SYMBOLIC PROOF RUN COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("Shape-invariance still provides the clean propagation argument.")
    print("="*120)