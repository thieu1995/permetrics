#!/usr/bin/env python
# Created by "Thieu" at 10:13, 23/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## 1. Import packages, classes
## 2. Create object
## 3. From object call function and use

from permetrics.classification import ClassificationMetric

## For integer labels or categorical labels
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

evaluator = ClassificationMetric(y_true, y_pred)

## Call specific function inside object, each function has 2 names like below

print(evaluator.f1_score())
print(evaluator.F1S(average="micro"))
print(evaluator.f1_score(average="macro"))
print(evaluator.F1S(average="weighted"))
