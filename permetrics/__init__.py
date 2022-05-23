# !/usr/bin/env python
# Created by "Thieu" at 11:23, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# ## 1. Import packages, classes
# ## 2. Create object
# ## 3. From object calls function and use
#
# import numpy as np
# from permetrics.regression import RegressionMetric
# from permetrics.classification import ClassificationMetric
#
###### Regression performance
# y_true = np.array([3, -0.5, 2, 7, 5, 6])
# y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])
#
# evaluator = RegressionMetric(y_true, y_pred, decimal=5)
#
# ## Get the result of any function you want to
# rmse = evaluator.RMSE()
# mse = evaluator.MSE()
# mae = evaluator.MAE()
# print(f"RMSE: {rmse}, MSE: {mse}, MAE: {mae}")
#
#
###### Classification performance
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
#
# evaluator = ClassificationMetric(y_true, y_pred, decimal=5)
#
# ## Get the result of any function you want to
# print(evaluator.f1_score())
# print(evaluator.F1S(average="micro"))
# print(evaluator.f1s(average="macro"))
# print(evaluator.f1s(average="weighted"))


__version__ = "1.3.0"

from . import regression
from . import classification
