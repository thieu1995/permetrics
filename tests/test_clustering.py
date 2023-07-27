#!/usr/bin/env python
# Created by "Thieu" at 10:02, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClusteringMetric

np.random.seed(42)


def test_internal_ClusteringMetric_class():
    # generate sample data
    X = np.random.uniform(-1, 10, size=(300, 6))
    y_pred = np.random.randint(0, 3, size=300)

    evaluator = ClusteringMetric(y_pred=y_pred, X=X, decimal=5)
    bhi1 = evaluator.ball_hall_index()
    bhi2 = evaluator.BHI(X=X, y_pred=y_pred)
    bhi3 = evaluator.BHI()

    assert bhi1 == bhi2
    assert bhi2 == bhi3


def test_external_ClusteringMetric_class():
    # generate sample data
    y_true = np.random.randint(0, 3, size=300)
    y_pred = np.random.randint(0, 3, size=300)

    evaluator = ClusteringMetric(y_true=y_true, y_pred=y_pred, decimal=5)
    prs1 = evaluator.precision_score()
    prs2 = evaluator.PrS(y_true=y_true, y_pred=y_pred)
    prs3 = evaluator.PrS()

    assert prs1 == prs2
    assert prs2 == prs3
