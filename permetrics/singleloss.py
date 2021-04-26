#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:59, 23/09/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# -------------------------------------------------------------------------------------------------------%

from numpy import round, abs, any, ndarray, isfinite, isnan, log
from numpy import max, sqrt, array, mean, dot, divide, arctan, sum, median, argsort, zeros, concatenate, var, std


class Metrics:
    """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    def __init__(self, y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        """
        if type(y_true) is ndarray and type(y_pred) is ndarray:
            y_true = y_true.astype('float64')
            y_pred = y_pred.astype('float64')  # x = x[~np.isnan(x)] can't remove if array is dtype object, only work with dtype float
            if y_true.ndim == 1 and y_pred.ndim == 1:
                ## Remove all Nan in y_pred
                y_true = y_true[~isnan(y_pred)]
                y_pred = y_pred[~isnan(y_pred)]
                ## Remove all Inf in y_pred
                self.y_true = y_true[isfinite(y_pred)]
                self.y_pred = y_pred[isfinite(y_pred)]

                self.y_true_clean = self.y_true[self.y_pred != 0]
                self.y_pred_clean = self.y_pred[self.y_pred != 0]
            else:
                if y_true.ndim == y_pred.ndim:
                    ## Remove all row with Nan in y_pred
                    y_true = y_true[~isnan(y_pred).any(axis=1)]
                    y_pred = y_pred[~isnan(y_pred).any(axis=1)]
                    ## Remove all row with Inf in y_pred
                    self.y_true = y_true[isfinite(y_pred).any(axis=1)]
                    self.y_pred = y_pred[isfinite(y_pred).any(axis=1)]

                    self.y_true_clean = self.y_true[~any(self.y_pred == 0, axis=1)]
                    self.y_pred_clean = self.y_pred[~any(self.y_pred == 0, axis=1)]
                else:
                    print("=====Failed! y_true and y_pred need to have same number of dimensions=======")
                    exit(0)
        else:
            print("=====Failed! y_true and y_pred need to be a ndarray=======")
            exit(0)

    def re_func(self, clean=False, decimal=3):
        """
            + Relative error. This function computes the relative error between two numbers,
            or for element between a pair of lists or numpy arrays.
            + Return:  double or list of doubles
                The relative error between actual and predicted
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        return round(y_pred / y_true - 1, decimal)

    def ae_func(self, clean=False, decimal=3):
        """
            + Absolute error. This function computes the absolute error between two numbers,
            or for element between a pair of lists or numpy arrays.
            + Return:  double or list of doubles
                The absolute error between actual and predicted
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        return round(abs(y_true) - abs(y_pred), decimal)

    def se_func(self, clean=False, decimal=3):
        """
            + Squared error. This function computes the squared error between two numbers,
            or for element between a pair of lists or numpy arrays.
            + Return:  double or list of doubles
                The squared error between actual and predicted
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        return round((y_true - y_pred)**2, decimal)

    def sle_func(self, clean=False, decimal=3):
        """
            + Squared log error. This function computes the squared log error between two numbers,
            or for element between a pair of lists or numpy arrays.
            + Return:  double or list of doubles
                The squared log error between actual and predicted
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        return round((log(y_true + 1) - log(y_pred + 1)) ** 2, decimal)

    def ll_func(self, clean=True, decimal=3):
        """
            + Log likelihood: This function computes the log likelihood between two numbers,
            or for element between a pair of lists or numpy arrays.
            + Return:  double or list of doubles
                The log likelihood between actual and predicted
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        score = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        return round(score, decimal)

    RE = re = re_func
    AE = ae = ae_func
    SE = se = se_func
    SLE = sle = sle_func
    LL = ll = ll_func

