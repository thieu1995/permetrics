#!/usr/bin/env python
# Created by "Thieu" at 10:00, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClassificationMetric
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def data():
    cm1 = ClassificationMetric(y_true=np.array([0, 1, 0, 0, 1, 0]), y_pred=np.array([0, 1, 0, 0, 0, 1]))

    # Example one-hot encoded y_true and y_pred
    y_true = np.array([[0, 1, 0],  # Class 1
                       [1, 0, 0],  # Class 0
                       [0, 0, 1],  # Class 2
                       [0, 1, 0],  # Class 1
                       [0, 0, 1]])  # Class 2
    y_pred = np.array([[0.1, 0.8, 0.1],  # Predicted probabilities for Class 1, Class 0, Class 2
                       [0.7, 0.2, 0.1],
                       [0.2, 0.3, 0.5],
                       [0.3, 0.6, 0.1],
                       [0.1, 0.2, 0.7]])
    cm2 = ClassificationMetric(y_true=y_true, y_pred=y_pred)

    y_true = np.array([0, 1, 2, 0, 2])  # Class 2
    y_pred = np.array([[0.1, 0.8, 0.1],  # Predicted probabilities for Class 1, Class 0, Class 2
                       [0.7, 0.2, 0.1],
                       [0.2, 0.3, 0.5],
                       [0.3, 0.6, 0.1],
                       [0.1, 0.2, 0.7]])
    cm3 = ClassificationMetric(y_true=y_true, y_pred=y_pred)
    return cm1, cm2, cm3


def test_PS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.PS(average=avg)
            assert isinstance(res, outs[idx])


def test_NPV(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.NPV(average=avg)
            assert isinstance(res, outs[idx])


def test_RS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.RS(average=avg)
            assert isinstance(res, outs[idx])


def test_AS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.AS(average=avg)
            assert isinstance(res, outs[idx])


def test_F1S(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.F1S(average=avg)
            assert isinstance(res, outs[idx])


def test_F2S(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.F2S(average=avg)
            assert isinstance(res, outs[idx])


def test_FBS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.FBS(average=avg)
            assert isinstance(res, outs[idx])


def test_SS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.SS(average=avg)
            assert isinstance(res, outs[idx])


def test_MCC(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.MCC(average=avg)
            assert isinstance(res, outs[idx])


def test_HS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.HS(average=avg)
            assert isinstance(res, outs[idx])


def test_LS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.LS(average=avg)
            assert isinstance(res, outs[idx])


def test_CKS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.CKS(average=avg)
            assert isinstance(res, outs[idx])


def test_JSI(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.JSI(average=avg)
            assert isinstance(res, outs[idx])


def test_GMS(data):
    avg_paras = [None, "macro", "micro", "weighted"]
    outs = (dict, float, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.GMS(average=avg)
            assert isinstance(res, outs[idx])


def test_GINI(data):
    for cm in data:
        res = cm.GINI()
        assert isinstance(res, float)


def test_CEL(data):
    for cm in data:
        res = cm.CEL()
        assert isinstance(res, float)


def test_HL(data):
    for cm in data:
        res = cm.HL()
        assert isinstance(res, float)


def test_KLDL(data):
    for cm in data:
        res = cm.KLDL()
        assert isinstance(res, float)


def test_BSL(data):
    for cm in data:
        res = cm.BSL()
        assert isinstance(res, float)


def test_ROC(data):
    avg_paras = [None, "macro", "weighted"]
    outs = (dict, float, float)

    for idx, avg in enumerate(avg_paras):
        for cm in data:
            res = cm.ROC(average=avg)
            assert isinstance(res, outs[idx])
