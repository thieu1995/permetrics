#!/usr/bin/env python
# Created by "Thieu" at 10:00, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClassificationMetric

np.random.seed(42)


def test_ClassificationMetric_class():
    y_true = [0, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 0, 1]

    evaluator = ClassificationMetric(y_true, y_pred, decimal=5)

    mcc1 = evaluator.matthews_correlation_coefficient()
    mcc2 = evaluator.MCC()
    mcc3 = evaluator.MCC(y_true, y_pred)

    assert mcc1 == mcc2
    assert mcc2 == mcc3
