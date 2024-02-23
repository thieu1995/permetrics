#!/usr/bin/env python
# Created by "Thieu" at 09:58, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import RegressionMetric
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def data():
    rm1 = RegressionMetric(y_true=np.array([3, -0.5, 2, 7, 5]), y_pred=np.array([2.5, 0.0, 2, 8, 5]))
    rm2 = RegressionMetric(y_true=np.array([[3, 1], [-0.5, 2], [2, 3], [7, 4], [5, 5], [6, 6]]),
                           y_pred=np.array([[2.5, 0.5], [0.0, 1.5], [2, 2], [8, 3], [5, 4], [3, 5]]))
    return rm1, rm2


def test_EVS(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.EVS()
    assert isinstance(res, (float))
    # ND
    res = rm2.EVS(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.EVS(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.EVS(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_ME(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.ME()
    assert isinstance(res, (float))
    # ND
    res = rm2.ME(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.ME(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.ME(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MBE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MBE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MBE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MBE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MBE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MAE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MAE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MAE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MAE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MAE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MSE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MSE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MSE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MSE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MSE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_RMSE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.RMSE()
    assert isinstance(res, (float))
    # ND
    res = rm2.RMSE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.RMSE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.RMSE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MSLE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MSLE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MSLE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MSLE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MSLE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MedAE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MedAE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MedAE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MedAE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MedAE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MRE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MRE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MRE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MRE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MRE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MPE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MPE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MPE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MPE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MPE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MAPE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MAPE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MAPE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MAPE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MAPE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_SMAPE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.SMAPE()
    assert isinstance(res, (float))
    # ND
    res = rm2.SMAPE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.SMAPE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.SMAPE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MAAPE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MAAPE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MAAPE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MAAPE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MAAPE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_MASE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.MASE()
    assert isinstance(res, (float))
    # ND
    res = rm2.MASE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.MASE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.MASE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_NSE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.NSE()
    assert isinstance(res, (float))
    # ND
    res = rm2.NSE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.NSE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.NSE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_NNSE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.NNSE()
    assert isinstance(res, (float))
    # ND
    res = rm2.NNSE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.NNSE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.NNSE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_WI(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.WI()
    assert isinstance(res, (float))
    # ND
    res = rm2.WI(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.WI(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.WI(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_PCC(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.PCC()
    assert isinstance(res, (float))
    # ND
    res = rm2.PCC(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.PCC(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.PCC(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_APCC(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.APCC()
    assert isinstance(res, (float))
    # ND
    res = rm2.APCC(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.APCC(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.APCC(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_RSQ(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.RSQ()
    assert isinstance(res, (float))
    # ND
    res = rm2.RSQ(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.RSQ(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.RSQ(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_CI(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.CI()
    assert isinstance(res, (float))
    # ND
    res = rm2.CI(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.CI(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.CI(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_COD(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.COD()
    assert isinstance(res, (float))
    # ND
    res = rm2.COD(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.COD(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.COD(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_ACOD(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.ACOD(X_shape=(6, 100))
    assert isinstance(res, (float))
    # ND
    res = rm2.ACOD(multi_output=None, X_shape=(6, 100))
    assert isinstance(res, (float))
    res = rm2.ACOD(multi_output="raw_values", X_shape=(6, 100))
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.ACOD(multi_output=(0.2, 0.7), X_shape=(6, 100))
    assert isinstance(res, float)


def test_DRV(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.DRV()
    assert isinstance(res, (float))
    # ND
    res = rm2.DRV(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.DRV(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.DRV(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_KGE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.KGE()
    assert isinstance(res, (float))
    # ND
    res = rm2.KGE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.KGE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.KGE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_PCD(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.PCD()
    assert isinstance(res, (float))
    # ND
    res = rm2.PCD(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.PCD(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.PCD(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_CE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.CE()
    assert isinstance(res, (float))
    # ND
    res = rm2.CE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.CE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.CE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_KLD(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.KLD()
    assert isinstance(res, (float))
    # ND
    res = rm2.KLD(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.KLD(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.KLD(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_JSD(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.JSD()
    assert isinstance(res, (float))
    # ND
    res = rm2.JSD(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.JSD(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.JSD(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_VAF(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.VAF()
    assert isinstance(res, (float))
    # ND
    res = rm2.VAF(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.VAF(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.VAF(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_RAE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.RAE()
    assert isinstance(res, (float))
    # ND
    res = rm2.RAE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.RAE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.RAE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_A10(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.A10()
    assert isinstance(res, (float))
    # ND
    res = rm2.A10(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.A10(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.A10(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_A20(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.A20()
    assert isinstance(res, (float))
    # ND
    res = rm2.A20(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.A20(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.A20(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_A30(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.A30()
    assert isinstance(res, (float))
    # ND
    res = rm2.A30(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.A30(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.A30(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_NRMSE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.NRMSE()
    assert isinstance(res, (float))
    # ND
    res = rm2.NRMSE(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.NRMSE(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.NRMSE(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_RSE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.RSE(n_paras=5)                            # 5 data samples
    assert isinstance(res, (float))
    # ND
    res = rm2.RSE(n_paras=6, multi_output=None)         # 6 data samples
    assert isinstance(res, (float))
    res = rm2.RSE(n_paras=6, multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.RSE(n_paras=6, multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_COV(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.COV()
    assert isinstance(res, (float))
    # ND
    res = rm2.COV(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.COV(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.COV(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_COR(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.COR()
    assert isinstance(res, (float))
    # ND
    res = rm2.COR(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.COR(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.COR(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_EC(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.EC()
    assert isinstance(res, (float))
    # ND
    res = rm2.EC(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.EC(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.EC(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_OI(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.OI()
    assert isinstance(res, (float))
    # ND
    res = rm2.OI(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.OI(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.OI(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_CRM(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.CRM()
    assert isinstance(res, (float))
    # ND
    res = rm2.CRM(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.CRM(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.CRM(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_GINI(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.GINI()
    assert isinstance(res, (float))
    # ND
    res = rm2.GINI(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.GINI(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.GINI(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_GINI_WIKI(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.GINI_WIKI()
    assert isinstance(res, (float))
    # ND
    res = rm2.GINI_WIKI(multi_output=None)
    assert isinstance(res, (float))
    res = rm2.GINI_WIKI(multi_output="raw_values")
    assert isinstance(res, (list, tuple, np.ndarray))
    res = rm2.GINI_WIKI(multi_output=(0.2, 0.7))
    assert isinstance(res, float)


def test_RE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.RE()
    assert isinstance(res, (list, tuple, np.ndarray))
    # ND
    res = rm2.RE()
    assert isinstance(res, (list, tuple, np.ndarray))


def test_AE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.AE()
    assert isinstance(res, (list, tuple, np.ndarray))
    # ND
    res = rm2.AE()
    assert isinstance(res, (list, tuple, np.ndarray))


def test_SE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.SE()
    assert isinstance(res, (list, tuple, np.ndarray))
    # ND
    res = rm2.SE()
    assert isinstance(res, (list, tuple, np.ndarray))


def test_SLE(data):
    rm1, rm2 = data[0], data[1]
    # 1D
    res = rm1.SLE()
    assert isinstance(res, (list, tuple, np.ndarray))
    # ND
    res = rm2.SLE()
    assert isinstance(res, (list, tuple, np.ndarray))
