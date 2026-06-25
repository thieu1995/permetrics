#!/usr/bin/env python
# Created by "Thieu" at 19:16, 24/06/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from permetrics import RegressionMetric

evaluator = RegressionMetric()

test_cases = [
    (480, 500),
    ([480, ], 500),
    ([10.0, 20.0], [10.0, 20.0]),                  # Khớp hoàn hảo - 0.0
    ([100.0], [90.0]),          # Lệch thường   -  "Tính toán bình thường"
    ([0.0, 0.0], [0.0, 0.0]),                      # Cả hai bằng 0 (Edge case)      , 0.0
    ([0.0], [10.0]),                             # Một bên bằng 0 (Max error)       , 200.0
    ([10.0], [0.0]),                             # Một bên bằng 0 ngược lại     , 200.0
    ([-5.0], [5.0]),                      # Số âm xem mẫu số xử lý ra sao       , "Test số âm"
    ([1e-9], [0.0]),                 # Số cực hạn gần 0     , "Test số siêu nhỏ"
    ([[500, 20]], [[480, 10]]),
    ([[10, 100], [20, 200]], [[10, 90], [20, 110]]),
    ([[1e-9, 100], [20, 200]], [[10, 90], [20, 110]]),
]

for y_true, y_pred in test_cases:
    print("==================\n")
    # e1 = evaluator.normalized_gini_coefficient(y_true, y_pred)
    # print(e1)
    # e2 = evaluator.residual_gini_index(y_true, y_pred)
    # print(e2)
    # e3 = evaluator.SMAPE(y_true, y_pred)
    # print(e3)
    # e4 = evaluator.SMAPE_NP(y_true, y_pred)
    # print(e4)
    # e5 = evaluator.SMAPE_S(y_true, y_pred)
    # print(e5)
    # e6 = evaluator.SMAPE_S_P(y_true, y_pred)
    # print(e6)
    e7 = evaluator.NRMSE(y_true, y_pred, normalization="mean")
    print(e7)
    e8= evaluator.NRMSE(y_true, y_pred, normalization="range")
    print(e8)
    e9 = evaluator.NRMSE(y_true, y_pred, normalization="std")
    print(e9)
    e10 = evaluator.NRMSE(y_true, y_pred, normalization="iqr")
    print(e10)


    # mse = evaluator.MSE(y_true, y_pred)
    # mae = evaluator.MAE(y_true, y_pred)
    # print(f"MSE: {mse}, MAE: {mae}")
