#!/usr/bin/env python
# Created by "Thieu" at 18:17, 22/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
from permetrics import ClusteringMetric
import permetrics.utils.cluster_util as cut
from sklearn.metrics import silhouette_score as sk_si
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

np.random.seed(100)


def generate_dataset(num_samples, num_features, num_clusters, cluster_std):
    centroids = np.random.randn(num_clusters, num_features)
    labels = np.random.randint(0, num_clusters, num_samples)
    data = centroids[labels] + np.random.randn(num_samples, num_features) * cluster_std
    return data, centroids, labels


def silhouette_score_ver1(X=None, y_pred=None):
    ## Slow and cost RAM - can't run with 50k
    ## 30k samples -> 90% RAM, 186 seconds
    dm = distance_matrix(X, X)
    res = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        a = np.mean(dm[i, y_pred == y_pred[i]])  # Cohesion
        b_values = [np.mean(dm[i, y_pred == label]) for label in np.unique(y_pred) if label != y_pred[i]]
        b = np.min(b_values) if len(b_values) > 0 else 0  # Separation
        res[i] = (b - a) / max(a, b)
    return np.mean(res)


def silhouette_score_ver2(X=None, y_pred=None, multi_output=False, force_finite=True, finite_value=-1.):
    ## Fast but cost RAM
    ## 30K samples -> 90% RAM, 71 seconds
    unique_clusters = np.unique(y_pred)
    if len(unique_clusters) == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Silhouette Index is undefined when y_pred has only 1 cluster.")
    num_clusters = len(unique_clusters)
    num_points = len(X)
    # Precompute pairwise distances
    pairwise_distances_matrix = cdist(X, X)
    a_values = np.zeros(num_points)
    b_values = np.zeros(num_points)
    for i in range(num_clusters):
        mask_i = y_pred == unique_clusters[i]
        mask_i_indices = np.where(mask_i)[0]
        a_values_i = np.sum(pairwise_distances_matrix[mask_i_indices][:, mask_i_indices], axis=1) / np.sum(mask_i)
        a_values[mask_i_indices] = a_values_i
        b_values_i = np.min([
            np.sum(pairwise_distances_matrix[mask_i_indices][:, y_pred == unique_clusters[j]], axis=1) / np.sum(y_pred == unique_clusters[j])
            for j in range(num_clusters) if j != i], axis=0)
        b_values[mask_i_indices] = b_values_i
    results = (b_values - a_values) / np.maximum(a_values, b_values)
    if multi_output:
        return results
    return np.mean(results)


def silhouette_score_ver3(data_points, cluster_assignments):
    ## Fast but can't run with 50k samples
    ## 30k samples, RAM 90%, and 54.8 seconds
    unique_clusters = np.unique(cluster_assignments)
    num_clusters = len(unique_clusters)
    num_points = len(data_points)

    # Precompute pairwise distances
    pairwise_distances_matrix = cdist(data_points, data_points)

    a_values = np.zeros(num_points)
    b_values = np.zeros(num_points)

    for i in range(num_clusters):
        mask_i = cluster_assignments == unique_clusters[i]
        mask_i_indices = np.where(mask_i)[0]

        a_values_i = np.sum(pairwise_distances_matrix[mask_i_indices][:, mask_i_indices], axis=1) / np.sum(mask_i)
        a_values[mask_i_indices] = a_values_i

        b_values_i = np.min([
            np.sum(pairwise_distances_matrix[mask_i_indices][:, cluster_assignments == unique_clusters[j]], axis=1) /
            np.sum(cluster_assignments == unique_clusters[j])
            for j in range(num_clusters) if j != i
        ], axis=0)
        b_values[mask_i_indices] = b_values_i

    silhouette_scores = (b_values - a_values) / np.maximum(a_values, b_values)
    overall_silhouette_score = np.mean(silhouette_scores)

    return overall_silhouette_score, silhouette_scores


# Example usage
num_samples = 20000
num_features = 10
num_clusters = 15
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

# Calculate SSE using the optimized function

t4 = time.perf_counter()
s4 = sk_si(data, labels)
print("Res1: ", s4, time.perf_counter() - t4)

cm = ClusteringMetric(y_pred=labels, X=data)
t5 = time.perf_counter()
# s5 = cut.calculate_silhouette_index(data, labels)
s5 = cm.silhouette_index()
print("Res2: ", s5, time.perf_counter() - t5)

t6 = time.perf_counter()
s6 = silhouette_score_ver1(data, labels)
print("Res3: ", s6, time.perf_counter() - t6)

t8 = time.perf_counter()
s8 = silhouette_score_ver2(data, labels)
print("Res4: ", s8, time.perf_counter() - t8)

time01 = time.perf_counter()
ch_score = silhouette_score_ver3(data, labels)
print("Res5:", ch_score, time.perf_counter() - time01)
