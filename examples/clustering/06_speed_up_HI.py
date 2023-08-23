#!/usr/bin/env python
# Created by "Thieu" at 18:17, 22/08/2023 ----------%                                                                               
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


def hartigans_index(data, labels, centroids):
    num_clusters = len(np.unique(labels))
    h = 0.0
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_data = data[cluster_indices]
        cluster_centroid = centroids[i]

        distances_within_cluster = cut.cdist(cluster_data, [cluster_centroid], metric='euclidean') ** 2
        sum_distances_within_cluster = np.sum(distances_within_cluster)

        other_centroids = np.delete(centroids, i, axis=0)
        closest_other_centroid_index = np.argmin(np.linalg.norm(cluster_centroid - other_centroids, axis=1))
        closest_other_centroid = other_centroids[closest_other_centroid_index]

        distances_to_closest_other_cluster = cut.cdist(cluster_data, [closest_other_centroid], metric='euclidean') ** 2
        sum_distances_to_closest_other_cluster = np.sum(distances_to_closest_other_cluster)

        h += sum_distances_within_cluster / sum_distances_to_closest_other_cluster
    return h


num_samples = 10000000
num_features = 2
num_clusters = 5
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

time03 = time.perf_counter()
s3 = hartigans_index(data, labels, centroids)
print("res: ", s3, time.perf_counter() - time03)

time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
res = cm.hartigan_index()
print("res: ", res, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_hartigan_index(data, labels)
print("res: ", s3, time.perf_counter() - time03)
