#!/usr/bin/env python
# Created by "Thieu" at 18:49, 12/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClassificationMetric
from sklearn.metrics import brier_score_loss


def brier_score_loss2(y_true, y_pred):
    num_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
    if num_classes == 1:  # Binary classification
        return np.mean((y_true - y_pred) ** 2)
    else:  # Multi-class classification
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))


# Binary classification example
y_true_binary = np.array([0, 1, 1, 0])  # True binary labels
y_pred_binary = np.array([0.3, 0.7, 0.9, 0.2])  # Predicted probabilities
cu = ClassificationMetric()
print(cu.brier_score_loss(y_true_binary, y_pred_binary))
print(brier_score_loss2(y_true_binary, y_pred_binary))
print(brier_score_loss(y_true_binary, y_pred_binary))

# Multi-Class Classification Example
y_true_multiclass = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # True class labels (one-hot encoded)
y_pred_multiclass = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.2, 0.1, 0.7]])  # Predicted class probabilities
print(cu.brier_score_loss(y_true_multiclass, y_pred_multiclass))
print(brier_score_loss2(y_true_multiclass, y_pred_multiclass))
# print(brier_score_loss(y_true_multiclass, y_pred_multiclass))
# Scikit Learn library can't even calculate in this case.
