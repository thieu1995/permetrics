#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:00, 19/07/2020                                                        %
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

y_true2 = array([3, -0.5, 2, 7])
y_pred2 = array([2.5, 0.0, 2, 9])

### C1. Using OOP style - very powerful when calculating multiple metrics
obj1 = Metrics(y_true, y_pred)  # Pass the data here
result = obj1.mean_absolute_percentage_error(clean=True, decimal=5)
print(f"1-D array, OOP style: {result}")

### C2. Using functional style
obj2 = Metrics()
result = obj2.mean_absolute_percentage_error(clean=True, decimal=5, y_true=y_true2, y_pred=y_pred2)  # Pass the data here, remember the keywords (y_true, y_pred)
print(f"1-D array, Functional style: {result}")

## > 1-D array - Multi-dimensional Array
y_true = array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = array([[0, 2], [-1, 2], [8, -5]])

multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
obj3 = Metrics(y_true, y_pred)
for multi_output in multi_outputs:
    result = obj3.mean_absolute_percentage_error(clean=True, multi_output=multi_output, decimal=5)
    print(f"n-D array, OOP style: {result}")