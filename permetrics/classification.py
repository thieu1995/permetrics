#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:29, 23/09/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# -------------------------------------------------------------------------------------------------------%


from numpy import max, round, sqrt, abs, mean, dot, divide, arctan, sum, any, median, log, var, std
from numpy import ndarray, array, isfinite, isnan, argsort, zeros, concatenate


class Metrics:
    """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    def __init__(self, y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        """
        self.onedim = False
        if type(y_true) is ndarray and type(y_pred) is ndarray:
            y_true = y_true.astype('float64')
            y_pred = y_pred.astype('float64')  # x = x[~np.isnan(x)] can't remove if array is dtype object, only work with dtype float
            if y_true.ndim == 1 and y_pred.ndim == 1:
                self.onedim = True
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

    def __multi_output_result__(self, result=None, multi_output=None, decimal=None):
        if multi_output is None:
            return round(mean(result), decimal)
        elif isinstance(multi_output, (tuple, list, set)):
            weights = array(multi_output)
            if self.y_true.ndim != len(weights):
                print("==========Failed! Multi-output weights has different length with y_true==============")
                exit(0)
            return round(dot(result, multi_output), decimal)
        elif isinstance(multi_output, ndarray) and len(multi_output) == self.y_true.ndim:
            return round(dot(result, multi_output), decimal)
        elif multi_output == "raw_values":
            return round(result, decimal)
        else:
            print("==========Failed! Multi-output not supported==============")
            exit(0)

    def mll_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Mean Log Likelihood (MLL)
            https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/elementwise.py#L235
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean

        if self.onedim:
            score = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
            return round(mean(score), decimal)
        else:
            score = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
            return self.__multi_output_result__(mean(score, axis=0), multi_output, decimal)

    MLL = mll = mll_func