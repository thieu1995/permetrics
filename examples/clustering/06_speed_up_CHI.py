#!/usr/bin/env python
# Created by "Thieu" at 18:17, 22/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
from permetrics import ClusteringMetric
import permetrics.utils.cluster_util as cut
from sklearn.metrics import calinski_harabasz_score as sk_chs


def generate_dataset(num_samples, num_features, num_clusters, cluster_std):
    centroids = np.random.randn(num_clusters, num_features)
    labels = np.random.randint(0, num_clusters, num_samples)
    data = centroids[labels] + np.random.randn(num_samples, num_features) * cluster_std
    return data, centroids, labels


def calinski_harabasz_score(data, labels, centroids):
    n_samples, n_features = data.shape
    n_clusters = centroids.shape[0]

    # Calculate the overall mean of the data
    overall_mean = np.mean(data, axis=0)

    # Calculate the between-cluster variance
    between_var = 0.0
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_size = np.sum(cluster_mask)
        cluster_mean = np.mean(data[cluster_mask], axis=0)
        between_var += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)

    # Calculate the within-cluster variance
    within_var = 0.0
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_mean = np.mean(data[cluster_mask], axis=0)
        within_var += np.sum(np.sum((data[cluster_mask] - cluster_mean) ** 2, axis=1))

    # Calculate the CH Index
    ch_index = (between_var / within_var) * ((n_samples - n_clusters) / (n_clusters - 1))
    return ch_index


def calinski_harabasz_score(data, labels, centroids):
    n_samples, n_features = data.shape
    n_clusters = centroids.shape[0]

    overall_mean = np.mean(data, axis=0)

    # Calculate between-cluster variance and cluster sizes
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    cluster_means = np.array([np.mean(data[labels == i], axis=0) for i in range(n_clusters)])
    between_var = np.sum(cluster_sizes * np.sum((cluster_means - overall_mean) ** 2, axis=1))

    # Calculate within-cluster variance
    within_var = np.sum((data - cluster_means[labels]) ** 2)

    # Calculate the CH Index
    ch_index = (between_var / within_var) * ((n_samples - n_clusters) / (n_clusters - 1))
    return ch_index


# Example usage
num_samples = 10000000
num_features = 2
num_clusters = 5
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

# Calculate SSE using the optimized function
time01 = time.perf_counter()
ch_score = calinski_harabasz_score(data, labels, centroids)
print("Calinski-Harabasz Score:", ch_score, time.perf_counter() - time01)

time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
sse02 = cm.calinski_harabasz_index()
print("Res: ", sse02, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_calinski_harabasz_index(data, labels)
print("Res: ", s3, time.perf_counter() - time03)

t4 = time.perf_counter()
s4 = sk_chs(data, labels)
print("Res: ", s4, time.perf_counter() - t4)
