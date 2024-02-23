#!/usr/bin/env python
# Created by "Thieu" at 11:35, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## This is modern and better way to use metrics
## You only need to pass y_true, y_pred one time when creating metric object,
## After that, you can get the value of any metrics without passing y_true, y_pred

## 1. Import packages, classes
## 2. Create object
## 3. From object call function and use

import numpy as np
from permetrics.classification import ClassificationMetric

y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

evaluator = ClassificationMetric(y_true, y_pred)

## Get the result of any function you want to

hamming_score = evaluator.hamming_score()
mcc = evaluator.matthews_correlation_coefficient()
specificity = evaluator.specificity_score()

print(f"HL: {hamming_score}, MCC: {mcc}, specificity: {specificity}")

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
#     HS = hamming_score
#     LS = lift_score
#     CKS = cohen_kappa_score
#     JSI = JSC = jaccard_similarity_coefficient = jaccard_similarity_index
#     GMS = g_mean_score
#     GINI = gini_index
#     ROC = AUC = RAS = roc_auc_score
