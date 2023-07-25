#!/usr/bin/env python
# Created by "Thieu" at 14:44, 05/02/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics.classification import ClassificationMetric

## For integer labels or categorical labels
y_true = [0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

# y_true = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 0])
# y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1])

# y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
# y_pred = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])

# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

# y_true = [["cat", "ant"], ["cat", "cat"], ["ant", "bird"], ["bird", "bird"]]
# y_pred = [["ant", "ant"], ["cat", "cat"], ["ant", "cat"], ["bird", "ant"]]

# cm = ClassificationMetric(y_true, y_pred, decimal=5)
# print(cm.jaccard_similarity_index(average=None))
# print(cm.jaccard_similarity_coefficient(average="micro"))
# print(cm.jsi(average="macro"))
# print(cm.jsc(average="weighted"))

# cm = ClassificationMetric(y_true, y_pred, decimal=5)
# print(cm.gini_index())
# print(cm.cks(average="micro"))
# print(cm.CKS(average="macro"))
# print(cm.CKS(average="weighted"))

# print(cm.mcc(average=None))
# print(cm.mcc(average="micro"))
# print(cm.mcc(average="macro"))
# print(cm.mcc(average="weighted"))

# Example true labels and predicted scores for a 3-class problem
# y_true = np.array([0, 1, 2, 1, 2, 0, 0, 1])
# y_score = np.array([[0.8, 0.1, 0.1],
#                    [0.2, 0.5, 0.3],
#                    [0.1, 0.3, 0.6],
#                    [0.3, 0.7, 0.0],
#                    [0.4, 0.3, 0.3],
#                    [0.6, 0.2, 0.2],
#                    [0.9, 0.1, 0.0],
#                    [0.1, 0.8, 0.1]])
#
# # y_true = [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
# # y_score = [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
#
# cm = ClassificationMetric(y_true, y_pred, decimal=5)
# print(cm.roc_auc_score(y_true, y_score, average="weighted"))


cm = ClassificationMetric(y_true, y_pred, decimal=5)
print(cm.gini_index(average=None))
print(cm.GINI(average="macro"))
print(cm.gini(average="weighted"))

