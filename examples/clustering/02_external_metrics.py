#!/usr/bin/env python
# Created by "Thieu" at 06:32, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## Better to use OOP to call all of the available functions
## External metrics: Need y_true and y_pred and has suffix as score

import numpy as np
from permetrics import ClusteringMetric
from sklearn.datasets import make_blobs

# generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
y_pred = np.random.randint(0, 4, size=300)

evaluator = ClusteringMetric(y_true=y_true, y_pred=y_pred, decimal=5)

print(evaluator.get_metrics_by_list_names(["MIS", "NMIS", "RaS", "FMS", "HS", "CS", "VMS", "PrS", "ReS", "FmS",
                                           "CDS", "HGS", "JS", "KS", "MNS", "PhS", "RTS", "RRS", "SS1S", "SS2S"]))

# MIS = mutual_info_score
# NMIS = normalized_mutual_info_score
# RaS = rand_score
# FMS = fowlkes_mallows_score
# HS = homogeneity_score
# CS = completeness_score
# VMS = v_measure_score
# PrS = precision_score
# ReS = recall_score
# FmS = f_measure_score
# CDS = czekanowski_dice_score
# HGS = hubert_gamma_score
# JS = jaccard_score
# KS = kulczynski_score
# MNS = mc_nemar_score
# PhS = phi_score
# RTS = rogers_tanimoto_score
# RRS = russel_rao_score
# SS1S = sokal_sneath1_score
# SS2S = sokal_sneath2_score
