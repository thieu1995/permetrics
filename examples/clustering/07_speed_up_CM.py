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


def compute_confusion_matrix_old_version(y_true, y_pred, normalize=False):
    """
    Computes the confusion matrix for a clustering problem given the true labels and the predicted labels.
    http://cran.nexr.com/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    """
    n = len(y_true)
    yy = yn = ny = nn = 0.
    for i in range(n - 1):
        y_true_diff = y_true[i + 1:] == y_true[i]
        y_pred_diff = y_pred[i + 1:] == y_pred[i]
        yy += np.sum(y_true_diff & y_pred_diff)
        yn += np.sum(y_true_diff & ~y_pred_diff)
        ny += np.sum(~y_true_diff & y_pred_diff)
        nn += np.sum(~y_true_diff & ~y_pred_diff)
    res = np.array([yy, yn, ny, nn], dtype=np.int64)
    if normalize:
        return res / np.sum(res)
    return res


num_samples = 100000
num_features = 4
num_clusters = 8
cluster_std = 0.5

data, centroids, labels = generate_dataset(num_samples, num_features, num_clusters, cluster_std)


time02 = time.perf_counter()
cm = ClusteringMetric(y_true=labels, y_pred=labels, X=data)
res = cut.compute_confusion_matrix(y_true=labels, y_pred=labels)
print("CM 1: ", res, time.perf_counter() - time02 )

time03 = time.perf_counter()
s3 = compute_confusion_matrix_old_version(y_true=labels, y_pred=labels)
print("CM 2: ", s3, time.perf_counter() - time03)
