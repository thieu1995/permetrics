#!/usr/bin/env python
# Created by "Thieu" at 18:17, 22/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
from permetrics import ClusteringMetric
import permetrics.utils.cluster_util as cut
from sklearn.datasets import make_blobs

np.random.seed(100)


def generate_dataset(n_samples, n_features, n_clusters, random_state=42):
    X, y_true, centers = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters,
                           cluster_std=1.0, random_state=random_state, return_centers=True)
    y_pred = np.random.randint(0, n_clusters, n_samples)
    return X, y_true, y_pred, centers


num_samples = 10000
num_features = 2
num_clusters = 7
cluster_std = 0.5

data, y_true, y_pred, centers = generate_dataset(num_samples, num_features, num_clusters)


# time02 = time.perf_counter()
# cm = ClusteringMetric(y_true=y_true, y_pred=y_pred, decimal=10)
# res = cm.czekanowski_dice_score()
# print("res: ", res, time.perf_counter() - time02 )
#
# time03 = time.perf_counter()
# s3 = cut.calculate_czekanowski_dice_score(y_true, y_pred, decimal=10)
# print("res: ", s3, time.perf_counter() - time03)


# time02 = time.perf_counter()
# cm = ClusteringMetric(y_true=y_true, y_pred=y_pred, decimal=10)
# res = cm.hubert_gamma_score()
# print("res: ", res, time.perf_counter() - time02 )
#
# time03 = time.perf_counter()
# s3 = cut.calculate_hubert_gamma_score(y_true, y_pred, decimal=10)
# print("res: ", s3, time.perf_counter() - time03)


time02 = time.perf_counter()
cm = ClusteringMetric(y_true=y_true, y_pred=y_pred, decimal=10)
res = cm.jaccard_score()
print("res: ", res, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = cut.calculate_jaccard_score(y_true, y_pred, decimal=10)
print("res: ", s3, time.perf_counter() - time03)

