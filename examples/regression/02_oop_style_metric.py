#!/usr/bin/env python
# Created by "Thieu" at 11:35, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## This is modern and better way to use metrics
## You only need to pass y_true, y_pred one time when creating metric object,
## After that, you can get the value of any metrics without passing y_true, y_pred

## 1. Import packages, classes
## 2. Create object
## 3. From object call function and use

import numpy as np
from permetrics.regression import RegressionMetric

y_true = np.array([3, -0.5, 2, 7, 5, 6])
y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

evaluator = RegressionMetric(y_true, y_pred, decimal=5)

## Get the result of any function you want to

rmse = evaluator.RMSE()
mse = evaluator.MSE()
mae = evaluator.MAE()

print(f"RMSE: {rmse}, MSE: {mse}, MAE: {mae}")
