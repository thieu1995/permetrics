#!/usr/bin/env python
# Created by "Thieu" at 11:37, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## 1. Import packages, classes
## 2. Create object
## 3. From object call function and use

import numpy as np
from permetrics.regression import RegressionMetric

y_true = np.array([3, -0.5, 2, 7, 5, 6])
y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

evaluator = RegressionMetric()

## Call specific function inside object, each function has 3 names like below

rmse_1 = evaluator.RMSE(y_true, y_pred)
rmse_2 = evaluator.rmse(y_true, y_pred)
rmse_3 = evaluator.root_mean_squared_error(y_true, y_pred)
print(f"RMSE: {rmse_1}, {rmse_2}, {rmse_3}")


# EVS = evs = explained_variance_score
# ME = me = max_error
# MAE = mae = mean_absolute_error
# MSE = mse = mean_squared_error
# RMSE = rmse = root_mean_squared_error
# MSLE = msle = mean_squared_log_error
# MedAE = medae = median_absolute_error
# MRE = mre = mean_relative_error
# MAPE = mape = mean_absolute_percentage_error
# SMAPE = smape = symmetric_mean_absolute_percentage_error
# MAAPE = maape = mean_arctangent_absolute_percentage_error
# MASE = mase = mean_absolute_scaled_error
# NSE = nse = nash_sutcliffe_efficiency
# NNSE = nnse = normalized_nash_sutcliffe_efficiency
# WI = wi = willmott_index
# R = r = PCC = pcc = pearson_correlation_coefficient
# R2s = r2s = pearson_correlation_coefficient_square
# CI = ci = confidence_index
# R2 = r2 = coefficient_of_determination
# DRV = drv = deviation_of_runoff_volume
# KGE = kge = kling_gupta_efficiency
# GINI = gini = gini_coefficient
# GINI_WIKI = gini_wiki = gini_coefficient_wiki
# PCD = pcd = prediction_of_change_in_direction
# CE = ce = cross_entropy
# KLD = kld = kullback_leibler_divergence
# JSD = jsd = jensen_shannon_divergence
# VAF = vaf = variance_accounted_for
# RAE = rae = relative_absolute_error
# A10 = a10 = a10_index
# A20 = a20 = a20_index
# NRMSE = nrmse = normalized_root_mean_square_error
# RSE = rse = residual_standard_error

# RE = re = single_relative_error
# AE = ae = single_absolute_error
# SE = se = single_squared_error
# SLE = sle = single_squared_log_error
