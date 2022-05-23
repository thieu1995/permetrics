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

## 3.1 Call specific function inside object, each function has 3 names like below

ps1 = evaluator.precision_score(y_true, y_pred, decimal=5)
ps2 = evaluator.ps(y_true, y_pred)
ps3 = evaluator.PS(y_true, y_pred)
print(f"Precision: {ps1}, {ps2}, {ps3}")

recall = evaluator.recall_score(y_true, y_pred)
accuracy = evaluator.accuracy_score(y_true, y_pred)
print(f"recall: {recall}, accuracy: {accuracy}")
