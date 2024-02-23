#!/usr/bin/env python
# Created by "Thieu" at 09:22, 02/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClusteringMetric
from sklearn.datasets import make_blobs

# generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
y_pred_rand = []
for idx in range(0, len(y_true)):
    y_pred_rand.append(np.random.choice(list(set(range(0, 4)) - {idx})))
temp = [
    y_true.copy(), y_pred_rand, np.random.randint(0, 4, size=300),
    np.random.randint(0, 2, 300), np.random.randint(0, 6, 300),
    np.ones((300,)), np.zeros((300,))
]
for idx in range(7):
    evaluator = ClusteringMetric(y_pred=temp[idx], X=X)
    print(evaluator.hartigan_index())

# print(evaluator.get_metrics_by_list_names(["BHI", "XBI", "DBI", "BRI", "KDI", "DRI", "DI", "CHI",
#                                           "LDRI", "LSRI", "SI", "SSEI", "MSEI", "DHI", "BI", "RSI", "DBCVI", "HI"]))

# BHI = ball_hall_index
# XBI = xie_beni_index
# DBI = davies_bouldin_index
# BRI = banfeld_raftery_index
# KDI = ksq_detw_index
# DRI = det_ratio_index
# DI = dunn_index
# CHI = calinski_harabasz_index
# LDRI = log_det_ratio_index
# LSRI = log_ss_ratio_index
# SI = silhouette_index
# SSEI = sum_squared_error_index
# MSEI = mean_squared_error_index
# DHI = duda_hart_index
# BI = beale_index
# RSI = r_squared_index
# DBCVI = density_based_clustering_validation_index
# HI = hartigan_index
