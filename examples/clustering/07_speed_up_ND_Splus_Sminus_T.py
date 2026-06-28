#!/usr/bin/env python
# Created by "Thieu" at 00:09, 28/06/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
from permetrics import ClusteringMetric
import permetrics.utils.cluster_util as cut

np.random.seed(100)


def generate_dataset(num_samples, num_features, num_clusters, cluster_std):
    centroids = np.random.randn(num_clusters, num_features)
    labels = np.random.randint(0, num_clusters, num_samples)
    data = centroids[labels] + np.random.randn(num_samples, num_features) * cluster_std
    return data, centroids, labels


def compute_nd_splus_sminus_t_old_version(y_true=None, y_pred=None):
    """concordant_discordant"""
    n_samples = len(y_true)
    nd = n_samples * (n_samples - 1) / 2
    s_plus = 0.  # Number of concordant comparisons
    t = 0.  # Number of comparisons of two pairs of objects with same cluster labels
    for idx in range(n_samples - 1):
        t += np.sum((y_true[idx] == y_true[idx + 1:]) & (y_pred[idx] == y_pred[idx + 1:]))
        s_plus += np.sum((y_true[idx] == y_true[idx + 1:]) & (y_pred[idx] == y_pred[idx + 1:]))
        s_plus += np.sum((y_true[idx] != y_true[idx + 1:]) & (y_pred[idx] != y_pred[idx + 1:]))
    s_minus = nd - s_plus       # Number of discordant comparisons
    return nd, s_plus, s_minus, t


num_samples = 100000
num_features = 4
num_clusters = 8
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)

time02 = time.perf_counter()
res1 = cut.compute_nd_splus_sminus_t(y_true=labels, y_pred=labels)
print("Res 1: ", res1, time.perf_counter() - time02 )

time03 = time.perf_counter()
res2 = compute_nd_splus_sminus_t_old_version(y_true=labels, y_pred=labels)
print("Res 2: ", res2, time.perf_counter() - time03)
