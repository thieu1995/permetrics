#!/usr/bin/env python
# Created by "Thieu" at 11:37, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## To reduce coding time for using multiple metrics. There are few ways to do it with permetrics
## We have to use OOP style

import numpy as np
from permetrics.classification import ClassificationMetric

y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

evaluator = ClassificationMetric(y_true, y_pred, decimal=5)

## Define list of metrics you want to use


## 1. Get list metrics by using loop
list_metrics = ["PS", "RS", "LS", "SS"]
list_results = []
for metric in list_metrics:
    list_results.append( evaluator.get_metric_by_name(metric) )
print(list_results)


## 2. Get list metrics by using function
dict_result_2 = evaluator.get_metrics_by_list_names(list_metrics)
print(dict_result_2)


## 3. Get list metrics by using function and parameters
dict_metrics = {
    "PS": {"average": "micro"},
    "RS": {"average": "macro"},
    "LS": None,
    "SS": {"average": "weighted"},
}
dict_result_3 = evaluator.get_metrics_by_dict(dict_metrics)
print(dict_result_3)

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
