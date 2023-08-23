#!/usr/bin/env python
# Created by "Thieu" at 18:17, 22/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
from permetrics import ClusteringMetric
import permetrics.utils.cluster_util as cut
from scipy.spatial.distance import cdist

np.random.seed(100)


def generate_dataset(num_samples, num_features, num_clusters, cluster_std):
    centroids = np.random.randn(num_clusters, num_features)
    labels = np.random.randint(0, num_clusters, num_samples)
    data = centroids[labels] + np.random.randn(num_samples, num_features) * cluster_std
    return data, centroids, labels


def xie_beni_index(data, labels, centroids):
    num_clusters = len(np.unique(labels))  # Number of clusters
    wgss = np.sum(np.min(cdist(data, centroids, metric='euclidean'), axis=1) ** 2)
    list_dist = []
    for k in range(num_clusters):
        for k0 in range(k + 1, num_clusters):
            list_dist.append(np.sum((centroids[k] - centroids[k0]) ** 2))
    C = (wgss / np.min(list_dist)) / len(data)
    return C


num_samples = 10000000
num_features = 2
num_clusters = 5
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)

time03 = time.perf_counter()
s3 = xie_beni_index(data, labels, centroids)
print("XBI 1: ", s3, time.perf_counter() - time03)

time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
res = cm.xie_beni_index()
print("XBI 2: ", res, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_xie_beni_index(data, labels)
print("XBI 3: ", s3, time.perf_counter() - time03)
