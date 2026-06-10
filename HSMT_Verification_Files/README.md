# HSMT Verification Suite v9.7

High-precision numeric verification tools for **Hierarchical Shell-Manifold Theory (HSMT)**.

## Overview

This repository contains a complete, reproducible verification framework supporting the HSMT foundational paper (v9.7). It provides:

- High-precision numeric verification that the hypergeometric eigenfunctions satisfy the full octonionic Master Operator to high accuracy.
- Testing of the refined multifractal holographic projectors and graded channel operators.
- Quantitative evaluation of radial leakage effects from the N=5 layer on cosmological observables (H₀, S₈, and dynamical dark energy).

## Features

- High-precision arithmetic using `mpmath` (50 decimal places)
- Explicit residual norms for eigenfunction verification
- Separated static and dynamical contributions in the N=5 leakage model
- Clean, modular structure suitable for research use

## Requirements

```bash
pip install numpy scipy mpmath psutil
