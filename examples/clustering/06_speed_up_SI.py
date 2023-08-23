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


def silhouette_score(data_points, cluster_assignments):
    pairwise_distances = np.sqrt(((data_points[:, np.newaxis] - data_points) ** 2).sum(axis=2))

    def calculate_a(i, cluster_idx):
        return np.mean(pairwise_distances[i, cluster_assignments == cluster_idx])

    def calculate_b(i):
        return np.min([calculate_a(i, other_cluster) for other_cluster in np.unique(cluster_assignments) if other_cluster != cluster_assignments[i]])

    a_values = np.array([calculate_a(i, cluster_assignments[i]) for i in range(len(data_points))])
    b_values = np.array([calculate_b(i) for i in range(len(data_points))])

    silhouette_scores = (b_values - a_values) / np.maximum(a_values, b_values)
    overall_silhouette_score = np.mean(silhouette_scores)

    return overall_silhouette_score, silhouette_scores


def silhouette_score(data_points, cluster_assignments):
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
num_samples = 10000
num_features = 2
num_clusters = 5
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

# Calculate SSE using the optimized function
time01 = time.perf_counter()
ch_score, _ = silhouette_score(data, labels)
print("Res:", ch_score, time.perf_counter() - time01)

t4 = time.perf_counter()
s4 = sk_si(data, labels)
print("Res: ", s4, time.perf_counter() - t4)

time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
sse02 = cm.silhouette_index()
print("Res: ", sse02, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_silhouette_index(data, labels)
print("Res: ", s3, time.perf_counter() - time03)


