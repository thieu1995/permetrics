#!/usr/bin/env python
# Created by "Thieu" at 09:58, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import RegressionMetric

np.random.seed(42)


def test_RegressionMetric_class():
    y_true = np.array([3, -0.5, 2, 7, 5, 6])
    y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

    evaluator = RegressionMetric(y_true, y_pred, decimal=5)
    rmse1 = evaluator.RMSE()
    rmse2 = evaluator.root_mean_squared_error()
    assert rmse1 == rmse2
