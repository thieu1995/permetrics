#!/usr/bin/env python
# Created by "Thieu" at 11:34, 23/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics.classification import ClassificationMetric

y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

evaluator = ClassificationMetric(y_true, y_pred, decimal=5)
cm, imap, imap_count = evaluator.confusion_matrix()
print(cm)
print(imap)
print(imap_count)

print("======================================")

y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

evaluator = ClassificationMetric(y_true, y_pred, decimal=5)
cm, imap, imap_count = evaluator.confusion_matrix()
print(cm)
print(imap)
print(imap_count)

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
