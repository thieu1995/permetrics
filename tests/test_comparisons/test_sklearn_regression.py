#!/usr/bin/env python
# Created by "Thieu" at 18:21, 22/02/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score, mean_absolute_percentage_error

from permetrics import RegressionMetric


def is_close_enough(x1, x2, eps=1e-5):
    if abs(x1 - x2) <= eps:
        return True
    return False


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def data():
    y_true = np.array([3, -0.5, 2, 7, 5, 3, 4, -3, 10])
    y_pred = np.array([2.5, 0.0, 2, 8, 5, 2, 3.5, -4, 9])
    rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
    return y_true, y_pred, rm


def test_EVS(data):
    y_true, y_pred, rm = data
    res11 = rm.EVS()
    res12 = explained_variance_score(y_true, y_pred)
    assert is_close_enough(res11, res12)


def test_ME(data):
    y_true, y_pred, rm = data
    res11 = rm.ME()
    res12 = max_error(y_true, y_pred)
    assert is_close_enough(res11, res12)


def test_MAE(data):
    y_true, y_pred, rm = data
    res11 = rm.MAE()
    res12 = mean_absolute_error(y_true, y_pred)
    assert is_close_enough(res11, res12)


def test_MSE(data):
    y_true, y_pred, rm = data
    res11 = rm.MSE()
    res12 = mean_squared_error(y_true, y_pred)
    assert is_close_enough(res11, res12)


def test_MedAE(data):
    y_true, y_pred, rm = data
    res11 = rm.MedAE()
    res12 = median_absolute_error(y_true, y_pred)
    assert is_close_enough(res11, res12)


def test_R2(data):
    y_true, y_pred, rm = data
    res11 = rm.R2()
    res12 = r2_score(y_true, y_pred)
    assert is_close_enough(res11, res12)


def test_MAPE(data):
    y_true, y_pred, rm = data
    res11 = rm.MAPE()
    res12 = mean_absolute_percentage_error(y_true, y_pred)
    assert is_close_enough(res11, res12)
