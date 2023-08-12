#!/usr/bin/env python
# Created by "Thieu" at 15:30, 12/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.metrics import roc_auc_score
from permetrics import ClassificationMetric

# Example usage
y_true_binary = np.array([0, 1, 0, 1, 1])
y_pred_binary = np.array([0.9, 0.71, 0.6, 0.8, 0.7])

cu = ClassificationMetric()
print(cu.roc_auc_score(y_true_binary, y_pred_binary, average="weighted"))
print(roc_auc_score(y_true_binary, y_pred_binary))


# Example usage
y_true_multiclass = np.array([0, 1, 2, 1, 2])
y_pred_multiclass = np.array([[0.6, 0.2, 0.2],
                              [0.1, 0.7, 0.2],
                              [0.3, 0.4, 0.3],
                              [0.8, 0.1, 0.1],
                              [0.2, 0.6, 0.2]])

cu = ClassificationMetric()
print(cu.roc_auc_score(y_true_multiclass, y_pred_multiclass, average="macro"))
print(roc_auc_score(y_true_multiclass, y_pred_multiclass, multi_class="ovr"))


