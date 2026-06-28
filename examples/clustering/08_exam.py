#!/usr/bin/env python
# Created by "Thieu" at 19:58, 28/06/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics.clustering import ClusteringMetric


# ==============================================================================
# SCENARIO 1: Basic Evaluation
# ==============================================================================
print("--- 1. BASIC PURITY SCORE EXAMPLE ---")

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 1, 2, 2]

cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
pus_score = cm.PuS()
print(f"Purity Score: {pus_score:.4f}")  # Returns 0.8333 (5 out of 6 pure)

# ==============================================================================
# SCENARIO 2: Demonstrating the "Singleton Illusion"
# ==============================================================================
print("\n--- 2. THE SINGLETON ILLUSION ---")

# 100 completely random true classes
y_true_random = [0, 0, 1, 1, 2, 2]
# Model cheats by putting every point into its own separate cluster
y_pred_cheating = [10, 11, 12, 13, 14, 15]

cm_cheat = ClusteringMetric(y_true=y_true_random, y_pred=y_pred_cheating)
print(f"Cheating Model Purity: {cm_cheat.PuS()}")  # Returns 1.0 (Beware!)



# ==============================================================================
# SCENARIO 1: Basic Evaluation (Perfect vs Impure)
# ==============================================================================
print("--- 1. BASIC ENTROPY SCORE EXAMPLE ---")

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 0, 1, 1, 1]

cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
print(f"Perfect Clustering Entropy: {cm.EnS():.4f}")  # Returns 0.0000

# Introduce chaos (cluster 1 now has mixed classes)
y_pred_impure = [0, 0, 1, 1, 1, 0]
cm_impure = ClusteringMetric(y_true=y_true, y_pred=y_pred_impure)
print(f"Impure Clustering Entropy:  {cm_impure.EnS():.4f}")  # Returns higher value

# ==============================================================================
# SCENARIO 2: Maximum Uncertainty Benchmark
# ==============================================================================
print("\n--- 2. MAXIMUM UNCERTAINTY EXAMPLE ---")

# Every predicted cluster is a 50/50 coin flip of class 0 and class 1
cm_chaos = ClusteringMetric(y_true=[0, 1, 0, 1], y_pred=[0, 0, 1, 1])
print(f"Maximum Entropy Score: {cm_chaos.EnS():.4f}")  # Returns 2.0



# ==============================================================================
# SCENARIO 1: Basic Evaluation
# ==============================================================================
print("--- 1. BASIC TAU SCORE EXAMPLE ---")

y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 0, 1, 1, 1, 2]

cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
taus_score = cm.TauS()
print(f"Tau Score: {taus_score:.4f}")

# ==============================================================================
# SCENARIO 2: Verifying Peak Correlation on Identical Inputs
# ==============================================================================
print("\n--- 2. IDENTICAL PARTITION CHECK ---")

cm_perfect = ClusteringMetric(y_true=[0, 1, 2, 3], y_pred=[10, 20, 30, 40])
print(f"Perfect Match Tau Score: {cm_perfect.TauS()}")





# ==============================================================================
# SCENARIO 1: Basic Evaluation
# ==============================================================================
print("--- 1. BASIC GAMMA SCORE EXAMPLE ---")

y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 0, 1, 1, 1, 2]

cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
gas_score = cm.GAS()
print(f"Gamma Score: {gas_score:.4f}")

# ==============================================================================
# SCENARIO 2: Verifying the Rand Score Identity
# ==============================================================================
print("\n--- 2. RAND SCORE AFFINE TRANSFORMATION CHECK ---")

ras_score = cm.RaS()
derived_gas = (2 * ras_score) - 1

print(f"Direct GAS:  {gas_score}")
print(f"2*RaS - 1:   {derived_gas}")
print(f"Are they exact? {np.isclose(gas_score, derived_gas)}")

from permetrics.clustering import ClusteringMetric

# ==============================================================================
# SCENARIO 1: Basic Evaluation
# ==============================================================================
print("--- 1. BASIC G-PLUS SCORE EXAMPLE ---")

y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 0, 1, 1, 2, 2]

cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
print(f"Perfect Match GPS (Loss): {cm.GPS():.4f}")  # Returns 0.0000

# ==============================================================================
# SCENARIO 2: Verifying the Complement Identity
# ==============================================================================
print("\n--- 2. COMPLEMENT IDENTITY CHECK ---")

cm_bad = ClusteringMetric(y_true=[0, 0, 0], y_pred=[1, 2, 3])

print(f"Direct GPS:     {cm_bad.GPS()}")
print(f"1.0 - RaS:      {1.0 - cm_bad.RaS()}")
print(f"Are they exact? {np.isclose(cm_bad.GPS(), 1.0 - cm_bad.RaS())}")