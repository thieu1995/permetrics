#!/usr/bin/env python
# Created by "Thieu" at 18:10, 12/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClassificationMetric


def kl_divergence_loss_binary(y_true, y_pred):
    epsilon = 1e-10  # Small epsilon value to avoid division by zero
    y_true = np.clip(y_true, epsilon, 1 - epsilon)  # Clip true labels
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted probabilities
    loss = y_true * np.log(y_true / y_pred) + (1 - y_true) * np.log((1 - y_true) / (1 - y_pred))
    return np.mean(loss)

def kl_divergence_loss_multiclass(y_true, y_pred):
    epsilon = 1e-10  # Small epsilon value to avoid division by zero
    y_true = np.clip(y_true, epsilon, 1 - epsilon)  # Clip true labels
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted probabilities
    loss = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(loss)

# Binary classification example
y_true_binary = np.array([0, 1, 1, 0])  # True binary labels
y_pred_binary = np.array([0.3, 0.7, 0.9, 0.2])  # Predicted probabilities
binary_loss = kl_divergence_loss_binary(y_true_binary, y_pred_binary)
print("Binary Loss:", binary_loss)

# Multi-class classification example
y_true_multiclass = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # True class labels (one-hot encoded)
y_pred_multiclass = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.2, 0.1, 0.7]])  # Predicted class probabilities
multiclass_loss = kl_divergence_loss_multiclass(y_true_multiclass, y_pred_multiclass)
print("Multi-Class Loss:", multiclass_loss)

cu = ClassificationMetric(y_true_binary, y_pred_binary)
print(cu.kullback_leibler_divergence_loss())

cu = ClassificationMetric()
print(cu.kullback_leibler_divergence_loss(y_true_multiclass, y_pred_multiclass))
