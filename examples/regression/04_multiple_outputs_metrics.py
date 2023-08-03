#!/usr/bin/env python
# Created by "Thieu" at 11:37, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## Scikit-learn library is limited with multi-output metrics, but permetrics can produce multi-output for all of metrics

import numpy as np
from permetrics.regression import RegressionMetric

## This y_true and y_pred have 4 columns, 4 outputs
y_true = np.array([ [3, -0.5, 2, 7],
                    [5, 6, -0.3, 9],
                    [-11, 23, 8, 3.9] ])

y_pred = np.array([ [2.5, 0.0, 2, 8],
                    [5.2, 5.4, 0, 9.1],
                    [-10, 23, 8.2, 4] ])

evaluator = RegressionMetric(y_true, y_pred, decimal=5)

## 1. By default, all metrics can automatically return the multi-output results
# rmse = evaluator.RMSE()
# print(rmse)

## 2. If you want to take mean of all outputs, can set the parameter: multi-output = "mean"
# rmse_2 = evaluator.RMSE(multi_output="mean")
# print(rmse_2)

## 3. If you want a specific metric has more important than other, you can set weight for each output.
# rmse_3 = evaluator.RMSE(multi_output=[0.5, 0.05, 0.1, 0.35])
# print(rmse_3)


## Get multiple metrics with multi-output or single-output by parameters


## 1. Get list metrics by using list_names
list_metrics = ["RMSE", "MAE", "MSE"]
list_paras = [
    {"decimal": 3, "multi_output": "mean"},
    {"decimal": 4, "multi_output": [0.5, 0.2, 0.1, 0.2]},
    {"decimal": 5, "multi_output": "raw_values"}
]
dict_result_1 = evaluator.get_metrics_by_list_names(list_metrics, list_paras)
print(dict_result_1)


## 2. Get list metrics by using dict_metrics
dict_metrics = {
    "RMSE": {"decimal": 5, "multi_output": "mean"},
    "MAE": {"decimal": 4, "multi_output": "raw_values"},
    "MSE": {"decimal": 2, "multi_output": [0.5, 0.2, 0.1, 0.2]},
}
dict_result_2 = evaluator.get_metrics_by_dict(dict_metrics)
print(dict_result_2)

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
