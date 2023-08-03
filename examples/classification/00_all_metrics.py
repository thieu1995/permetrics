#!/usr/bin/env python
# Created by "Thieu" at 10:13, 23/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## 1. Import packages, classes
## 2. Create object
## 3. From object call function and use

import numpy as np
from permetrics.classification import ClassificationMetric

## For integer labels or categorical labels
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

evaluator = ClassificationMetric(y_true, y_pred, decimal=5)

## Call specific function inside object, each function has 2 names like below

print(evaluator.f1_score())
print(evaluator.F1S(average="micro"))
print(evaluator.f1_score(average="macro"))
print(evaluator.F1S(average="weighted"))

#     CM = confusion_matrix
#     PS = precision_score
#     NPV = negative_predictive_value
#     RS = recall_score
#     AS = accuracy_score
#     F1S = f1_score
#     F2S = f2_score
#     FBS = fbeta_score
#     SS = specificity_score
#     MCC = matthews_correlation_coefficient
#     HL = hamming_loss
#     LS = lift_score
#     CKS = cohen_kappa_score
#     JSI = JSC = jaccard_similarity_coefficient = jaccard_similarity_index
#     GMS = g_mean_score
#     GINI = gini_index
#     ROC = AUC = RAS = roc_auc_score
