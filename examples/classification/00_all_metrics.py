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

## Call specific function inside object, each function has 3 names like below

print(evaluator.f1_score())
print(evaluator.F1S(average="micro"))
print(evaluator.f1s(average="macro"))
print(evaluator.f1s(average="weighted"))

# PS = ps = precision_score
# NPV = npv = negative_predictive_value
# RS = rs = recall_score
# AS = accuracy_score
# F1S = f1s = f1_score
# F2S = f2s = f2_score
# FBS = fbs = fbeta_score
# SS = ss = specificity_score
# MCC = mcc = matthews_correlation_coefficient
# HL = hl = hamming_loss
# LS = ls = lift_score