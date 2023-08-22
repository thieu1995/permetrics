#!/usr/bin/env python
# Created by "Thieu" at 18:17, 22/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
from permetrics import ClusteringMetric
import permetrics.utils.cluster_util as cut


def generate_dataset(num_samples, num_features, num_clusters, cluster_std):
    centroids = np.random.randn(num_clusters, num_features)
    labels = np.random.randint(0, num_clusters, num_samples)
    data = centroids[labels] + np.random.randn(num_samples, num_features) * cluster_std
    return data, centroids, labels


def calculate_sse(data, centroids, labels):
    centroid_distances = centroids[labels]
    squared_distances = np.sum(np.square(data - centroid_distances), axis=1)
    sse = np.sum(squared_distances)
    return sse


num_samples = 10000000
num_features = 2
num_clusters = 5
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

# Calculate SSE using the optimized function
time01 = time.perf_counter()
sse = calculate_sse(data, centroids, labels)
print("Sum of Squared Errors:", sse, time.perf_counter() - time01)

time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
sse02 = cm.sum_squared_error_index()
print("SSE: ", sse02, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_sum_squared_error_index(data, labels)
print("SSE1: ", s3, time.perf_counter() - time03)
