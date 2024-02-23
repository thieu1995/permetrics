#!/usr/bin/env python
# Created by "Thieu" at 16:14, 07/02/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

from permetrics import ClassificationMetric


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def data():
    y_true1 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0])
    y_pred1 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    cm1 = ClassificationMetric(y_true=y_true1, y_pred=y_pred1)

    # Example one-hot encoded y_true and y_pred
    y_true2 = np.array([[0, 1, 0],  # Class 1
                        [1, 0, 0],  # Class 0
                        [0, 0, 1],  # Class 2
                        [0, 1, 0],  # Class 1
                        [0, 0, 1]])  # Class 2
    y_pred2 = np.array([[0.1, 0.8, 0.1],  # Predicted probabilities for Class 1, Class 0, Class 2
                        [0.7, 0.2, 0.1],
                        [0.2, 0.3, 0.5],
                        [0.3, 0.6, 0.1],
                        [0.1, 0.2, 0.7]])
    cm2 = ClassificationMetric(y_true=y_true2, y_pred=y_pred2)

    y_true3 = np.array([0, 1, 2, 0, 2])  # Class 2
    y_pred3 = np.array([[0.1, 0.8, 0.1],  # Predicted probabilities for Class 1, Class 0, Class 2
                        [0.7, 0.2, 0.1],
                        [0.2, 0.3, 0.5],
                        [0.3, 0.6, 0.1],
                        [0.1, 0.2, 0.7]])
    cm3 = ClassificationMetric(y_true=y_true3, y_pred=y_pred3)
    return (y_true1, y_pred1), (y_true2, y_pred2), (y_true3, y_pred3), cm1, cm2, cm3


def test_AS(data):
    (y_true1, y_pred1), (y_true2, y_pred2), (y_true3, y_pred3), cm1, cm2, cm3 = data
    res11 = cm1.PS(average="micro")
    res12 = accuracy_score(y_true1, y_pred1)
    assert res11 == res12

    # res21 = cm2.PS(average="micro")
    # res22 = accuracy_score(y_true2, y_pred2)      # ValueError: Classification metrics can't handle a mix of multiclass and continuous-multioutput targets
    # assert res21 == res22

    # res31 = cm3.PS(average="micro")
    # res32 = accuracy_score(y_true3, y_pred3)      # ValueError: Classification metrics can't handle a mix of multiclass and continuous-multioutput targets
    # assert res31 == res32

    # avg_paras = [None, "macro", "micro", "weighted"]
    # outs = (dict, float, float, float)
    #
    # for idx, avg in enumerate(avg_paras):
    #     for cm in data:
    #         res = cm.PS(average=avg)
    #         assert isinstance(res, outs[idx])


def test_F1S(data):
    (y_true1, y_pred1), (y_true2, y_pred2), (y_true3, y_pred3), cm1, cm2, cm3 = data
    res11 = cm1.F1S(average="micro")
    res12 = f1_score(y_true1, y_pred1, average="micro")
    assert res11 == res12

    res11 = cm1.F1S(average="macro")
    res12 = f1_score(y_true1, y_pred1, average="macro")
    assert res11 == res12


def test_FBS(data):
    (y_true1, y_pred1), (y_true2, y_pred2), (y_true3, y_pred3), cm1, cm2, cm3 = data
    res11 = cm1.FBS(average="micro", beta=1.5)
    res12 = fbeta_score(y_true1, y_pred1, average="micro", beta=1.5)
    assert res11 == res12

    res11 = cm1.FBS(average="macro", beta=2.0)
    res12 = fbeta_score(y_true1, y_pred1, average="macro", beta=2.0)
    assert res11 == res12


def test_PS(data):
    (y_true1, y_pred1), (y_true2, y_pred2), (y_true3, y_pred3), cm1, cm2, cm3 = data
    res11 = cm1.PS(average="micro")
    res12 = precision_score(y_true1, y_pred1, average="micro")
    assert res11 == res12

    res11 = cm1.PS(average="macro")
    res12 = precision_score(y_true1, y_pred1, average="macro")
    assert res11 == res12


def test_RS(data):
    (y_true1, y_pred1), (y_true2, y_pred2), (y_true3, y_pred3), cm1, cm2, cm3 = data
    res11 = cm1.RS(average="micro")
    res12 = recall_score(y_true1, y_pred1, average="micro")
    assert res11 == res12

    res11 = cm1.RS(average="macro")
    res12 = recall_score(y_true1, y_pred1, average="macro")
    assert res11 == res12
