#!/usr/bin/env python
# Created by "Thieu" at 11:36, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## This is traditional way to call a specific metric you want to use.
## Everytime, you want to use a function, you need to pass y_true and y_pred

## 1. Import packages, classes
## 2. Create object
## 3. From object call function and use

import numpy as np
from permetrics.classification import ClassificationMetric

y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

evaluator = ClassificationMetric()

## 3.1 Call specific function inside object, each function has 2 names like below

ps1 = evaluator.precision_score(y_true, y_pred, decimal=5)
ps2 = evaluator.PS(y_true, y_pred)
print(f"Precision: {ps1}, {ps2}")

recall = evaluator.recall_score(y_true, y_pred)
accuracy = evaluator.accuracy_score(y_true, y_pred)
print(f"recall: {recall}, accuracy: {accuracy}")

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
