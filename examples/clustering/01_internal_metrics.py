#!/usr/bin/env python
# Created by "Thieu" at 06:25, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## Better to use OOP to call all of the available functions
## Internal metrics: Need X and y_pred and has suffix as index

import numpy as np
from permetrics import ClusteringMetric
from sklearn.datasets import make_blobs

# generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
y_pred = np.random.randint(0, 4, size=300)

evaluator = ClusteringMetric(y_pred=y_pred, X=X, decimal=5)

print(evaluator.get_metrics_by_list_names(["BHI", "CHI", "XBI", "BRI", "DBI", "DRI", "DI", "KDI", "LDRI", "LSRI", "SI"]))

# BHI = ball_hall_index
# CHI = calinski_harabasz_index
# XBI = xie_beni_index
# BRI = banfeld_raftery_index
# DBI = davies_bouldin_index
# DRI = det_ratio_index
# DI = dunn_index
# KDI = ksq_detw_index
# LDRI = log_det_ratio_index
# LSRI = log_ss_ratio_index
# SI = silhouette_index
