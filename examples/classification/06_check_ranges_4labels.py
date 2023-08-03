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
