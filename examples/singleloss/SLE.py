#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:11, 23/09/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# -------------------------------------------------------------------------------------------------------%

from numpy import array
from permetrics.singleloss import Metrics

## 1-D array
y_true = array([3, -0.5, 2, 7])
y_pred = array([2.5, 0.0, 2, 8])

obj1 = Metrics(y_true, y_pred)
print(obj1.sle_func(clean=False, decimal=5))

## > 1-D array
y_true = array([[0.5, 1], [-0.5, 1], [7, 6]])
y_pred = array([[0, 2], [-0.5, 2], [8, 5]])

obj2 = Metrics(y_true, y_pred)
print(obj2.sle_func(clean=False, decimal=5))
