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
