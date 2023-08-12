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
# from permetrics import RegressionMetric, ClassificationMetric, ClusteringMetric
#
# ##### Regression performance
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
# ##### Classification performance
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
#
# evaluator = ClassificationMetric(y_true, y_pred, decimal=5)
#
# ## Get the result of any function you want to
# print(evaluator.f1_score())
# print(evaluator.F1S(average="micro"))
# print(evaluator.F1S(average="macro"))
# print(evaluator.F1S(average="weighted"))
#
#
# ##### Clustering performance
# X = np.random.uniform(-1, 10, size=(300, 5))
# y_true = np.random.randint(0, 4, size=300)
# y_pred = np.random.randint(0, 4, size=300)
#
# external_evaluator = ClusteringMetric(y_true=y_true, y_pred=y_pred, decimal=5)
# print(external_evaluator.mutual_info_score())
# print(external_evaluator.MIS())
#
# internal_evaluator = ClusteringMetric(y_pred=y_pred, X=X, decimal=5)
# print(internal_evaluator.banfeld_raftery_index())
# print(internal_evaluator.BRI())


__version__ = "1.4.3"

from .evaluator import Evaluator
from .classification import ClassificationMetric
from .regression import RegressionMetric
from .clustering import ClusteringMetric
