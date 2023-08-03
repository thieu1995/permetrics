#!/usr/bin/env python
# Created by "Thieu" at 11:37, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## To reduce coding time for using multiple metrics. There are few ways to do it with permetrics
## We have to use OOP style

import numpy as np
from permetrics.regression import RegressionMetric

y_true = np.array([3, -0.5, 2, 7, 5, 6])
y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

evaluator = RegressionMetric(y_true, y_pred, decimal=5)

## Define list of metrics you want to use


## 1. Get list metrics by using loop
list_metrics = ["RMSE", "MAE", "MAPE", "NSE"]
list_results = []
for metric in list_metrics:
    list_results.append( evaluator.get_metric_by_name(metric) )
print(list_results)


## 2. Get list metrics by using function
dict_result_2 = evaluator.get_metrics_by_list_names(list_metrics)
print(dict_result_2)


## 3. Get list metrics by using function and parameters
dict_metrics = {
    "RMSE": {"decimal": 5},
    "MAE": {"decimal": 4},
    "MAPE": None,
    "NSE": {"decimal": 3},
}
dict_result_3 = evaluator.get_metrics_by_dict(dict_metrics)
print(dict_result_3)

#     EVS = explained_variance_score
#     ME = max_error
#     MBE = mean_bias_error
#     MAE = mean_absolute_error
#     MSE = mean_squared_error
#     RMSE = root_mean_squared_error
#     MSLE = mean_squared_log_error
#     MedAE = median_absolute_error
#     MRE = MRB = mean_relative_bias = mean_relative_error
#     MPE = mean_percentage_error
#     MAPE = mean_absolute_percentage_error
#     SMAPE = symmetric_mean_absolute_percentage_error
#     MAAPE = mean_arctangent_absolute_percentage_error
#     MASE = mean_absolute_scaled_error
#     NSE = nash_sutcliffe_efficiency
#     NNSE = normalized_nash_sutcliffe_efficiency
#     WI = willmott_index
#     R = PCC = pearson_correlation_coefficient
#     AR = APCC = absolute_pearson_correlation_coefficient
#     RSQ = R2s = pearson_correlation_coefficient_square
#     CI = confidence_index
#     COD = R2 = coefficient_of_determination
#     ACOD = AR2 = adjusted_coefficient_of_determination
#     DRV = deviation_of_runoff_volume
#     KGE = kling_gupta_efficiency
#     PCD = prediction_of_change_in_direction
#     CE = cross_entropy
#     KLD = kullback_leibler_divergence
#     JSD = jensen_shannon_divergence
#     VAF = variance_accounted_for
#     RAE = relative_absolute_error
#     A10 = a10_index
#     A20 = a20_index
#     A30 = a30_index
#     NRMSE = normalized_root_mean_square_error
#     RSE = residual_standard_error
#     COV = covariance
#     COR = correlation
#     EC = efficiency_coefficient
#     OI = overall_index
#     CRM = coefficient_of_residual_mass
#     GINI = gini_coefficient
#     GINI_WIKI = gini_coefficient_wiki
#
#     RE = RB = single_relative_bias = single_relative_error
#     AE = single_absolute_error
#     SE = single_squared_error
#     SLE = single_squared_log_error
