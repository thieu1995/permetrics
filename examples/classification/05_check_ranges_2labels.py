#!/usr/bin/env python
# Created by "Thieu" at 14:13, 03/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics.classification import ClassificationMetric

y_true = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])

t1 = [
    y_true.copy(),
    1 - y_true,
    np.zeros(len(y_true)),
    np.ones(len(y_true)),
    np.random.randint(0, 2, len(y_true))
]
for idx in range(len(t1)):
    evaluator = ClassificationMetric(y_true, t1[idx])
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
