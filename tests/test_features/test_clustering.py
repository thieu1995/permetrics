#!/usr/bin/env python
# Created by "Thieu" at 00:15, 29/06/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import pytest

from permetrics import ClusteringMetric

# ==============================================================================
# GLOBAL SETUP
# ==============================================================================
np.random.seed(42)


INTERNAL_METRICS = ["BHI", "CHI", "XBI", "DBI", "BRI", "DRI", "KDI", "DI", "LDRI", "LSRI", "SI",
                    "SSEI", "MSEI", "DHI", "BI", "RSI", "HI"]
EXTERNAL_METRICS = [
    "MIS", "NMIS", "RaS", "ARS", "FMS", "HS", "CS", "VMS",
    "PrS", "ReS", "FS", "CDS", "HGS",
    "JS", "KS", "MNS", "PhS", "RTS", "RRS", "SS1S", "SS2S", "PuS", "EnS",
    "TauS", "GAS", "GPS"
]


# ==============================================================================
# FIXTURES: DATA GENERATORS
# ==============================================================================

@pytest.fixture(scope="module")
def normal_internal_data():
    X = np.random.uniform(-10, 10, size=(100, 5))
    y_pred = np.random.randint(0, 4, size=100)
    return X, y_pred


@pytest.fixture(scope="module")
def normal_external_data():
    y_true = np.random.randint(0, 5, size=200)
    y_pred = np.random.randint(0, 5, size=200)
    return y_true, y_pred


@pytest.fixture(scope="module")
def dbcv_data():
    X = np.random.rand(100, 2)
    y_pred = np.array([0] * 45 + [1] * 45 + [-1] * 10)
    return X, y_pred


# ==============================================================================
# 1. NORMAL SCENARIOS
# ==============================================================================

@pytest.mark.parametrize("metric", INTERNAL_METRICS)
def test_internal_metrics_normal(normal_internal_data, metric):
    X, y_pred = normal_internal_data
    cm = ClusteringMetric(X=X, y_pred=y_pred, force_finite=True)

    method = getattr(cm, metric)
    res = method()

    assert isinstance(res, float), f"{metric} must return a float"
    assert not np.isnan(res), f"{metric} returned NaN"


@pytest.mark.parametrize("metric", EXTERNAL_METRICS)
def test_external_metrics_normal(normal_external_data, metric):
    y_true, y_pred = normal_external_data
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred, force_finite=True)

    method = getattr(cm, metric)
    res = method()

    assert isinstance(res, float), f"{metric} must return a float"
    assert not np.isnan(res), f"{metric} returned NaN"


def test_dbcv_metric_normal(dbcv_data):
    X, y_pred = dbcv_data
    cm = ClusteringMetric(X=X, y_pred=y_pred, force_finite=True)
    res = cm.DBCVI(return_type="global")

    assert isinstance(res, float), "DBCV must return a float"
    assert -1.0 <= res <= 1.0, "DBCV must be bounded in [-1.0, 1.0]"


# ==============================================================================
# 2. EXTREME EDGE CASES
# ==============================================================================

EDGE_CASES_EXTERNAL = {
    "identical_partitions": ([0, 1, 2, 3], [0, 1, 2, 3]),
    "completely_different": ([0, 0, 1, 1], [2, 3, 4, 5]),
    "all_true_singletons": ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0]),
    "all_pred_singletons": ([0, 0, 0, 0, 0], [0, 1, 2, 3, 4]),
    "single_universal_cluster": ([0, 0, 0, 0], [0, 0, 0, 0]),
    "two_samples_only": ([0, 1], [0, 1]),
    "string_labels": (['cat', 'dog', 'cat'], ['cat', 'dog', 'dog'])
}


@pytest.mark.parametrize("edge_name, data", EDGE_CASES_EXTERNAL.items())
@pytest.mark.parametrize("metric", EXTERNAL_METRICS)
def test_external_metrics_edge_cases(edge_name, data, metric):
    y_true, y_pred = data
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred, force_finite=True)
    method = getattr(cm, metric)

    try:
        res = method()
        assert isinstance(res, float), f"Failed {metric} on {edge_name}: expected float"
        assert not np.isnan(res), f"Failed {metric} on {edge_name}: returned NaN"
    except Exception as e:
        pytest.fail(f"{metric} crashed on edge case '{edge_name}' with error: {str(e)}")


EDGE_CASES_INTERNAL = {
    "single_cluster": (np.random.rand(10, 2), [0] * 10),
    "all_singletons": (np.random.rand(5, 2), [0, 1, 2, 3, 4]),
    "zero_variance_data": (np.zeros((10, 3)), [0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
}


@pytest.mark.parametrize("edge_name, data", EDGE_CASES_INTERNAL.items())
@pytest.mark.parametrize("metric", INTERNAL_METRICS)
def test_internal_metrics_edge_cases(edge_name, data, metric):
    X, y_pred = data
    cm = ClusteringMetric(X=X, y_pred=y_pred, force_finite=True)
    method = getattr(cm, metric)

    try:
        res = method()
        assert isinstance(res, float), f"Failed {metric} on {edge_name}: expected float"
        assert not np.isnan(res), f"Failed {metric} on {edge_name}: returned NaN"
    except Exception as e:
        pytest.fail(f"{metric} crashed on edge case '{edge_name}' with error: {str(e)}")


# ==============================================================================
# 3. MATHEMATICAL IDENTITY & BOUNDARY VERIFICATIONS
# ==============================================================================

def test_mathematical_identities():
    """Verify known mathematical relationships between metrics."""
    y_true = [0, 0, 1, 1, 2, 2, 3]
    y_pred = [0, 1, 1, 2, 2, 2, 3]

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)

    # 1. Gamma = 2 * Rand - 1
    assert np.isclose(cm.GAS(), 2 * cm.RaS() - 1.0)

    # 2. G-Plus = 1 - Rand
    assert np.isclose(cm.GPS(), 1.0 - cm.RaS())

    # 3. Hierarchy Check
    assert cm.SS1S() <= cm.JS() <= cm.CDS()
    assert cm.RTS() <= cm.RaS() <= cm.SS2S()
