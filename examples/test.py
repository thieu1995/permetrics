#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:16, 26/02/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np


# Entropy
def entropy(p, q):
    return -np.sum(p * np.log(q.clip(1e-12, None)))


# Cross-entropy
def ce(pred, real):
    f_real, intervals = np.histogram(real, bins=len(np.unique(real)) - 1)
    intervals[0] = min(min(pred), min(real))
    intervals[-1] = max(max(pred), max(real))
    f_real = f_real / real.size
    f_pred = np.histogram(pred, bins=intervals)[0] / pred.size
    return entropy(f_real, f_pred)


# Kullback-Leibler divergence
def kldiv(pred, real):
    f_real, intervals = np.histogram(real, bins=len(np.unique(real)) - 1)
    intervals[0] = min(min(pred), min(real))
    intervals[-1] = max(max(pred), max(real))
    f_real = f_real / real.size
    f_pred = np.histogram(pred, bins=intervals)[0] / pred.size
    # kl = entropy(f_pred, f_real) - entropy(f_pred, f_pred)
    kl = entropy(f_real, f_pred) - entropy(f_real, f_real)
    return kl


pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10])
true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 1])

print(entropy(true, pred))
print(ce(pred, true))
print(kldiv(pred, true))

print(entropy(true, true))

# example of calculating the kl divergence between two mass functions
from math import log2


# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))


# define distributions
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate (P || Q)
kl_pq = kl_divergence(pred, true)
print('KL(P || Q): %.3f bits' % kl_pq)
# calculate (Q || P)
kl_qp = kl_divergence(true, pred)
print('KL(Q || P): %.3f bits' % kl_qp)