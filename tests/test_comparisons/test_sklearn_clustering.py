#!/usr/bin/env python
# Created by "Thieu" at 16:47, 23/02/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClusteringMetric
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, \
    adjusted_rand_score, rand_score, \
    completeness_score, homogeneity_score, v_measure_score, \
    fowlkes_mallows_score, calinski_harabasz_score, davies_bouldin_score
import pytest

np.random.seed(42)


def is_close_enough(x1, x2, eps=1e-5):
    if abs(x1 - x2) <= eps:
        return True
    return False


@pytest.fixture(scope="module")
def internal_data():
    # generate sample data
    X = np.random.uniform(-1, 10, size=(300, 6))
    y_pred = np.random.randint(0, 3, size=300)
    evaluator = ClusteringMetric(y_pred=y_pred, X=X, force_finite=True)
    return (X, y_pred), evaluator


@pytest.fixture(scope="module")
def external_data():
    # generate sample data
    y_true = np.random.randint(0, 3, size=300)
    y_pred = np.random.randint(0, 3, size=300)
    evaluator = ClusteringMetric(y_true=y_true, y_pred=y_pred, force_finite=True)
    return (y_true, y_pred), evaluator


def test_MIS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.MIS()
    res2 = mutual_info_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_NMIS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.NMIS()
    res2 = normalized_mutual_info_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_RaS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.RaS()
    res2 = rand_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_ARS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.ARS()
    res2 = adjusted_rand_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_CS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.CS()
    res2 = completeness_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_HS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.HS()
    res2 = homogeneity_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_VMS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.VMS()
    res2 = v_measure_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_FMS(external_data):
    (y_true, y_pred), cm = external_data
    res1 = cm.FMS()
    res2 = fowlkes_mallows_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_CHI(internal_data):
    (y_true, y_pred), cm = internal_data
    res1 = cm.CHI()
    res2 = calinski_harabasz_score(y_true, y_pred)
    assert is_close_enough(res1, res2)


def test_DBI(internal_data):
    (y_true, y_pred), cm = internal_data
    res1 = cm.DBI()
    res2 = davies_bouldin_score(y_true, y_pred)
    assert is_close_enough(res1, res2)
