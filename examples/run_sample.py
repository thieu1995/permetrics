#!/usr/bin/env python
# Created by "Thieu" at 19:16, 24/06/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from permetrics import RegressionMetric

evaluator = RegressionMetric()

# Giả sử bạn có 1 mẫu thực tế (y_true) và 1 mẫu dự đoán (y_pred)
# Ví dụ: Giá nhà thực tế là 500k$, mô hình dự đoán là 480k$

# BẮT BUỘC: Chuyển đổi thành mảng 2 chiều (1 sample, 1 feature)
y_true = 500
y_pred = 480

y_true = [500, ]
y_pred = (480, )

# y_true = (500, )
# y_pred = 480

# y_true = [500, 20]
# y_pred = [480, 10]

# y_true = [[500, 20]]
# y_pred = [[480, 10]]

# y_true = [[500, 20], [200, 30]]
# y_pred = [[480, 10], [180, 35]]

# e1 = evaluator.RMSE(y_true, y_pred)
e1 = evaluator.normalized_gini_coefficient(y_true, y_pred)
# e2 = evaluator.root_mean_squared_error(y_true, y_pred)
e2 = evaluator.residual_gini_index(y_true, y_pred)
print(f"{e1}, {e2}")

mse = evaluator.MSE(y_true, y_pred)
mae = evaluator.MAE(y_true, y_pred)
print(f"MSE: {mse}, MAE: {mae}")


# # Tính MSE và MAE
# mse = mean_squared_error(y_true, y_pred)
# mae = mean_absolute_error(y_true, y_pred)
#
# print(f"MSE cho 1 sample: {mse}")
# print(f"MAE cho 1 sample: {mae}")