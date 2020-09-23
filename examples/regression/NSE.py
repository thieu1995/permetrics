#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:02, 19/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import array
from permetrics.regression import Metrics

## 1-D array
y_true = array([3, -0.5, 2, 7])
y_pred = array([2.5, 0.0, 2, 8])

obj1 = Metrics(y_true, y_pred)
print(obj1.nse_func(clean=True, decimal=5))

## > 1-D array
y_true = array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = array([[0, 2], [-1, 2], [8, -5]])

multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
obj2 = Metrics(y_true, y_pred)
for multi_output in multi_outputs:
    print(obj2.nse_func(clean=True, multi_output=multi_output, decimal=5))
