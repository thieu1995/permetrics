#!/usr/bin/env python
# Created by "Thieu" at 16:03, 12/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClassificationMetric


def multiclass_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid numerical instability

    # Clip predicted probabilities to a minimum and maximum value
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute multi-class cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred), axis=1)

    # Take the average across samples
    loss = np.mean(loss)

    return loss

# Example usage
y_true_multiclass = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 0, 1]])
y_pred_multiclass = np.array([[0.2, 0.6, 0.2],
                              [0.7, 0.1, 0.2],
                              [0.3, 0.4, 0.3],
                              [0.8, 0.1, 0.1],
                              [0.4, 0.2, 0.4]])
multiclass_loss = multiclass_cross_entropy_loss(y_true_multiclass, y_pred_multiclass)
print("Multiclass Cross-Entropy Loss:", multiclass_loss)

cu = ClassificationMetric()
print(cu.crossentropy_loss(y_true_multiclass, y_pred_multiclass))
