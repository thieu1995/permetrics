#!/usr/bin/env python
# Created by "Thieu" at 10:02, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest

from permetrics import ClusteringMetric

np.random.seed(42)


@pytest.fixture(scope="module")
def internal_model():
    # generate sample data
    X = np.random.uniform(-1, 10, size=(300, 6))
    y_pred = np.random.randint(0, 3, size=300)
    evaluator = ClusteringMetric(y_pred=y_pred, X=X, force_finite=True)
    return evaluator


@pytest.fixture(scope="module")
def external_model():
    # generate sample data
    y_true = np.random.randint(0, 3, size=300)
    y_pred = np.random.randint(0, 3, size=300)
    evaluator = ClusteringMetric(y_true=y_true, y_pred=y_pred, force_finite=True)
    return evaluator


def test_BHI(internal_model):
    evaluator = internal_model
    res = evaluator.BHI()
    assert isinstance(res, (float, int))


def test_XBI(internal_model):
    evaluator = internal_model
    res = evaluator.XBI()
    assert isinstance(res, (float, int))


def test_DBI(internal_model):
    evaluator = internal_model
    res = evaluator.DBI()
    assert isinstance(res, (float, int))


def test_BRI(internal_model):
    evaluator = internal_model
    res = evaluator.BRI()
    assert isinstance(res, (float, int))


def test_KDI(internal_model):
    evaluator = internal_model
    res = evaluator.KDI()
    assert isinstance(res, (float, int))


def test_DRI(internal_model):
    evaluator = internal_model
    res = evaluator.DRI()
    assert isinstance(res, (float, int))


def test_DI(internal_model):
    evaluator = internal_model
    res = evaluator.DI()
    assert isinstance(res, (float, int))


def test_CHI(internal_model):
    evaluator = internal_model
    res = evaluator.CHI()
    assert isinstance(res, (float, int))


def test_LDRI(internal_model):
    evaluator = internal_model
    res = evaluator.LDRI()
    assert isinstance(res, (float, int))


def test_LSRI(internal_model):
    evaluator = internal_model
    res = evaluator.LSRI()
    assert isinstance(res, (float, int))


def test_SI(internal_model):
    evaluator = internal_model
    res = evaluator.SI()
    assert isinstance(res, (float, int))


def test_SSEI(internal_model):
    evaluator = internal_model
    res = evaluator.SSEI()
    assert isinstance(res, (float, int))


def test_MSEI(internal_model):
    evaluator = internal_model
    res = evaluator.MSEI()
    assert isinstance(res, (float, int))


def test_DHI(internal_model):
    evaluator = internal_model
    res = evaluator.DHI()
    assert isinstance(res, (float, int))


def test_BI(internal_model):
    evaluator = internal_model
    res = evaluator.BI()
    assert isinstance(res, (float, int))


def test_RSI(internal_model):
    evaluator = internal_model
    res = evaluator.RSI()
    assert isinstance(res, (float, int))


def test_DBCVI(internal_model):
    evaluator = internal_model
    res = evaluator.DBCVI()
    assert isinstance(res, (float, int))


def test_HI(internal_model):
    evaluator = internal_model
    res = evaluator.HI()
    assert isinstance(res, (float, int))


def test_MIS(external_model):
    evaluator = external_model
    res = evaluator.MIS()
    assert isinstance(res, (float, int))


def test_NMIS(external_model):
    evaluator = external_model
    res = evaluator.NMIS()
    assert isinstance(res, (float, int))


def test_RaS(external_model):
    evaluator = external_model
    res = evaluator.RaS()
    assert isinstance(res, (float, int))


def test_ARS(external_model):
    evaluator = external_model
    res = evaluator.ARS()
    assert isinstance(res, (float, int))


def test_FMS(external_model):
    evaluator = external_model
    res = evaluator.FMS()
    assert isinstance(res, (float, int))


def test_HS(external_model):
    evaluator = external_model
    res = evaluator.HS()
    assert isinstance(res, (float, int))


def test_CS(external_model):
    evaluator = external_model
    res = evaluator.CS()
    assert isinstance(res, (float, int))


def test_VMS(external_model):
    evaluator = external_model
    res = evaluator.VMS()
    assert isinstance(res, (float, int))


def test_PrS(external_model):
    evaluator = external_model
    res = evaluator.PrS()
    assert isinstance(res, (float, int))


def test_ReS(external_model):
    evaluator = external_model
    res = evaluator.ReS()
    assert isinstance(res, (float, int))


def test_FmS(external_model):
    evaluator = external_model
    res = evaluator.FmS()
    assert isinstance(res, (float, int))


def test_CDS(external_model):
    evaluator = external_model
    res = evaluator.CDS()
    assert isinstance(res, (float, int))


def test_HGS(external_model):
    evaluator = external_model
    res = evaluator.HGS()
    assert isinstance(res, (float, int))


def test_JS(external_model):
    evaluator = external_model
    res = evaluator.JS()
    assert isinstance(res, (float, int))


def test_KS(external_model):
    evaluator = external_model
    res = evaluator.KS()
    assert isinstance(res, (float, int))


def test_MNS(external_model):
    evaluator = external_model
    res = evaluator.MNS()
    assert isinstance(res, (float, int))


def test_PhS(external_model):
    evaluator = external_model
    res = evaluator.PhS()
    assert isinstance(res, (float, int))


def test_RTS(external_model):
    evaluator = external_model
    res = evaluator.RTS()
    assert isinstance(res, (float, int))


def test_RRS(external_model):
    evaluator = external_model
    res = evaluator.RRS()
    assert isinstance(res, (float, int))


def test_SS1S(external_model):
    evaluator = external_model
    res = evaluator.SS1S()
    assert isinstance(res, (float, int))


def test_SS2S(external_model):
    evaluator = external_model
    res = evaluator.SS2S()
    assert isinstance(res, (float, int))


def test_PuS(external_model):
    evaluator = external_model
    res = evaluator.PuS()
    assert isinstance(res, (float, int))


def test_ES(external_model):
    evaluator = external_model
    res = evaluator.ES()
    assert isinstance(res, (float, int))


def test_TS(external_model):
    evaluator = external_model
    res = evaluator.TS()
    assert isinstance(res, (float, int))


def test_GAS(external_model):
    evaluator = external_model
    res = evaluator.GAS()
    assert isinstance(res, (float, int))


def test_GPS(external_model):
    evaluator = external_model
    res = evaluator.GPS()
    assert isinstance(res, (float, int))
