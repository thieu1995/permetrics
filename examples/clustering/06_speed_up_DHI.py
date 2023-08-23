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


def calculate_duda_hart_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    # Find the unique cluster labels
    unique_labels = np.unique(y_pred)
    if len(unique_labels) == 1:
        if raise_error:
            raise ValueError("The Duda-Hart index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value

    # Calculate squared Euclidean distances between data points
    pairwise_squared_distances = np.sum((X[:, np.newaxis] - X) ** 2, axis=2)

    # Initialize the numerator and denominator for Duda index calculation
    intra_cluster_distances = 0
    inter_cluster_distances = 0

    for label in unique_labels:
        cluster_indices = np.where(y_pred == label)[0]
        other_cluster_indices = np.where(y_pred != label)[0]

        # Compute the sum of squared distances within the current cluster
        intra_cluster_distances += np.sum(pairwise_squared_distances[cluster_indices][:, cluster_indices])

        # Compute the sum of squared distances to other clusters
        inter_cluster_distances += np.sum(pairwise_squared_distances[cluster_indices][:, other_cluster_indices])

    # Calculate the Duda index
    result = intra_cluster_distances / inter_cluster_distances
    return np.round(result, decimal)


def calculate_duda_hart_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    # Find the unique cluster labels
    unique_labels = np.unique(y_pred)
    if len(unique_labels) == 1:
        if raise_error:
            raise ValueError("The Duda-Hart index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value

    n_samples, n_features = X.shape

    # Calculate squared Euclidean distances between data points
    pairwise_squared_distances = np.sum((X[:, np.newaxis] - X) ** 2, axis=2)
    pairwise_distances = np.sqrt(pairwise_squared_distances)

    # Initialize the numerator and denominator for Duda index calculation
    intra_cluster_distances = 0
    inter_cluster_distances = 0

    for label in unique_labels:
        cluster_indices = np.where(y_pred == label)[0]
        other_cluster_indices = np.where(y_pred != label)[0]

        # Compute the sum of squared distances within the current cluster
        intra_cluster_distances += np.mean(pairwise_distances[cluster_indices][:, cluster_indices])

        # Compute the sum of squared distances to other clusters
        inter_cluster_distances += np.mean(pairwise_distances[cluster_indices][:, other_cluster_indices])

    # Calculate the Duda index
    result = intra_cluster_distances / inter_cluster_distances
    return np.round(result, decimal)


num_samples = 5000
num_features = 2
num_clusters = 5
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

time03 = time.perf_counter()
s3 = calculate_duda_hart_index(data, labels)
print("res: ", s3, time.perf_counter() - time03)

time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
res = cm.duda_hart_index()
print("res: ", res, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_duda_hart_index(data, labels)
print("res: ", s3, time.perf_counter() - time03)
