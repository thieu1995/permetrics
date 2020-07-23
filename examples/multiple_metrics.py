#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 13:04, 23/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import array
from permetrics.regression import Metrics

## 1-D array
y_true = array([3, -0.5, 2, 7])
y_pred = array([2.5, 0.0, 2, 8])
obj1 = Metrics(y_true, y_pred)

## by name will use default parameters of function
print(obj1.get_metrics_by_name("RMSE", "MAE"))

## by list you can change parameters of function
print(obj1.get_metrics_by_list(["RMSE", "MAE", "MAPE", "SMAPE"]))


print("=================================")

## > 1-D array
y_true = array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = array([[0, 2], [-1, 2], [8, -5]])
obj2 = Metrics(y_true, y_pred)

multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]

for multi_output in multi_outputs:
    list_funcs = ["RMSE", "MAPE", "R2"]
    list_paras = [
        {"multi_output": multi_output, "decimal": 4},
        {"clean": True, "multi_output": multi_output, "decimal": 6},
        {"clean": True, "multi_output": multi_output, "decimal": 5},
    ]
    print(obj2.get_metrics_by_list(list_funcs, list_paras))
