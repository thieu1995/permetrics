#!/usr/bin/env python
# Created by "Thieu" at 17:46, 12/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClassificationMetric


# Example usage
y_true_binary = np.array([0, 1, 0, 1, 1])
y_pred_binary = np.array([0.2, 0.7, 0.3, 0.8, 0.4])

cu = ClassificationMetric()
print(cu.hinge_loss(y_true_binary, y_pred_binary))

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

print(cu.hinge_loss(y_true_multiclass, y_pred_multiclass))
