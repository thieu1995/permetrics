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


def compute_BGSS_old_version(X, labels):
    """
    The between-group dispersion BGSS or between-cluster variance
    """
    barycenters, overall_barycenter = cut.compute_barycenters(X, labels)
    n_clusters = len(barycenters)
    # Calculate the overall mean of the data
    overall_mean = np.mean(X, axis=0)

    # Calculate the between-cluster variance
    between_var = 0.0
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_size = np.sum(cluster_mask)
        cluster_mean = np.mean(X[cluster_mask], axis=0)
        between_var += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)
    return between_var


num_samples = 10000000
num_features = 10
num_clusters = 1000
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)

time02 = time.perf_counter()
res = cut.compute_BGSS(X=data, labels=labels)
print("CM 1: ", res, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = compute_BGSS_old_version(X=data, labels=labels)
print("CM 2: ", s3, time.perf_counter() - time03)
