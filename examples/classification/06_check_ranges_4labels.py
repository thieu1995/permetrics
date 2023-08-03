#!/usr/bin/env python
# Created by "Thieu" at 14:25, 03/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics.classification import ClassificationMetric

y_true = np.array([0, 1, 0, 2, 3, 1, 3, 2, 0, 1, 3, 2, 3, 1, 0, 2, 3, 1, 1, 3, 2, 0])
y_pred_rand = []
for idx in range(0, len(y_true)):
    y_pred_rand.append(np.random.choice(list(set(range(0, 4)) - {idx})))
t1 = [
    y_true.copy(),
    y_pred_rand,
    np.zeros(len(y_true)),
    np.ones(len(y_true)),
    np.random.randint(0, 4, len(y_true))
]
for idx in range(len(t1)):
    evaluator = ClassificationMetric(y_true, t1[idx], decimal=5)
    print(evaluator.gini_index())

#     CM = cm = confusion_matrix
#     PS = ps = precision_score
#     NPV = npv = negative_predictive_value
#     RS = rs = recall_score
#     AS = accuracy_score
#     F1S = f1s = f1_score
#     F2S = f2s = f2_score
#     FBS = fbs = fbeta_score
#     SS = ss = specificity_score
#     MCC = mcc = matthews_correlation_coefficient
#     HL = hl = hamming_loss
#     LS = ls = lift_score
#     CKS = cks = cohen_kappa_score
#     JSI = jsi = JSC = jsc = jaccard_similarity_coefficient = jaccard_similarity_index
#     GMS = gms = g_mean_score
#     GINI = gini = gini_index
#     ROC = AUC = RAS = roc = auc = ras = roc_auc_score
