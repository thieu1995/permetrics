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

evaluator = ClassificationMetric(y_true, y_pred, decimal=5)

## Get the result of any function you want to

hamming_loss = evaluator.hamming_loss()
mcc = evaluator.matthews_correlation_coefficient()
specificity = evaluator.specificity_score()

print(f"HL: {hamming_loss}, MCC: {mcc}, specificity: {specificity}")
