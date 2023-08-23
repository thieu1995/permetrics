#!/usr/bin/env python
# Created by "Thieu" at 18:17, 22/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
from permetrics import ClusteringMetric
import permetrics.utils.cluster_util as cut
from sklearn.metrics import davies_bouldin_score as sk_dbs

np.random.seed(100)


def generate_dataset(num_samples, num_features, num_clusters, cluster_std):
    centroids = np.random.randn(num_clusters, num_features)
    labels = np.random.randint(0, num_clusters, num_samples)
    data = centroids[labels] + np.random.randn(num_samples, num_features) * cluster_std
    return data, centroids, labels


def davies_bouldin_index(data, labels, centroids):
    num_clusters = len(centroids)
    cluster_distances = np.linalg.norm(centroids[:, np.newaxis, :] - centroids, axis=2)
    db_values = np.zeros(num_clusters)
    for i in range(num_clusters):
        intra_cluster_distances = np.linalg.norm(data[labels == i] - centroids[i], axis=1)
        avg_intra_cluster_distance = np.mean(intra_cluster_distances)
        max_ratio = 0
        for j in range(num_clusters):
            if i != j:
                inter_cluster_distance = cluster_distances[i, j]
                avg_intra_cluster_distance_j = np.mean(np.linalg.norm(data[labels == j] - centroids[j], axis=1))
                ratio = (avg_intra_cluster_distance + avg_intra_cluster_distance_j) / inter_cluster_distance
                max_ratio = max(max_ratio, ratio)
        db_values[i] = max_ratio
    return np.mean(db_values)


# Example usage
num_samples = 10000000
num_features = 2
num_clusters = 5
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

time01 = time.perf_counter()
ch_score = davies_bouldin_index(data, labels, centroids)
print("DBI 1:", ch_score, time.perf_counter() - time01)

time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
sse02 = cm.davies_bouldin_index()
print("DBI 2: ", sse02, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_davies_bouldin_index(data, labels)
print("DBI 3: ", s3, time.perf_counter() - time03)

t4 = time.perf_counter()
s4 = sk_dbs(data, labels)
print("DBI 4: ", s4, time.perf_counter() - t4)
