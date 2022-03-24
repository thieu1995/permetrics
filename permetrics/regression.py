#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:07, 18/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

# from numpy import max, round, sqrt, abs, mean, dot, divide, arctan, sum, any, median, log, var, std
# from numpy import ndarray, array, isfinite, isnan, argsort, zeros, concatenate, diff, sign
# from numpy import min, histogram, unique, where, logical_and

import numpy as np
from copy import deepcopy


class RegressionMetric:
    """
    This is class contains all regression metrics (for both regression and time-series problem)
    Extension of scikit-learn library:
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

    Notes
    ~~~~~
    + Some methods in scikit-learn can't calculate the multi-output metrics
    + Therefore, this class can calculate the multi-output metrics for all methods
    """

    EPSILON = 1e-10
    ACCEPTED_TYPE = (list, tuple, np.ndarray)

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point
            **kwargs ():
        """
        if kwargs is None: kwargs = {}
        self.__set_keyword_arguments(kwargs)
        self.y_true_original = y_true
        self.y_pred_original = y_pred
        self.y_true = deepcopy(y_true)
        self.y_pred = deepcopy(y_pred)
        self.decimal = decimal
        self.y_true_clean, self.y_pred_clean = None, None
        self.one_dim, self.already_clean = False, False

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __clean_data(self, y_true=None, y_pred=None):
        """
        Get clean data and additional information for latter use

        Args:
            y_true (tuple, list, np.ndarray):
            y_pred (tuple, list, np.ndarray):

        Returns:
            y_true: after remove all Nan and Inf values
            y_pred: after remove all Nan and Inf values
            y_true_clean: after remove all Nan, Inf and 0 values
            y_pred_clean: after remove all Nan, Inf and 0 values
            dim: number of dimension in y_true
        """
        if isinstance(y_true, self.ACCEPTED_TYPE) and isinstance(y_pred, self.ACCEPTED_TYPE):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            ## Remove all dimensions of size 1
            y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
            # x = x[~np.isnan(x)] can't remove if array is dtype object, only work with dtype float
            y_true = y_true.astype('float64')
            y_pred = y_pred.astype('float64')

            if y_true.ndim == y_pred.ndim == 1:
                ## Remove all Nan in y_pred
                y_true = y_true[~np.isnan(y_pred)]
                y_pred = y_pred[~np.isnan(y_pred)]
                ## Remove all Inf in y_pred
                y_true = y_true[np.isfinite(y_pred)]
                y_pred = y_pred[np.isfinite(y_pred)]
                y_true_clean = y_true[y_pred != 0]
                y_pred_clean = y_pred[y_pred != 0]
                return y_true, y_pred, y_true_clean, y_pred_clean, True
            elif y_true.ndim == y_pred.ndim > 1:
                ## Remove all row with Nan in y_pred
                y_true = y_true[~np.isnan(y_pred).any(axis=1)]
                y_pred = y_pred[~np.isnan(y_pred).any(axis=1)]
                ## Remove all row with Inf in y_pred
                y_true = y_true[np.isfinite(y_pred).all(axis=1)]
                y_pred = y_pred[np.isfinite(y_pred).all(axis=1)]
                y_true_clean = y_true[~np.any(y_pred == 0, axis=1)]
                y_pred_clean = y_pred[~np.any(y_pred == 0, axis=1)]
                return y_true, y_pred, y_true_clean, y_pred_clean, False
            else:
                print("Permetrics Error! y_true and y_pred need to have same number of dimensions.")
                exit(0)
        else:
            print("Permetrics Error! y_true and y_pred need to be a list, tuple or np.array.")
            exit(0)

    def __positive_data(self, y_true=None, y_pred=None, one_dim=False, positive_only=False):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            one_dim (bool): The number of dimensions in data
            positive_only (bool): Calculate metric based on positive values only or not.

        Returns:
            y_true_used: y_true with all positive values in computation process.
            y_pred_used: y_pred with all positive values in computation process
        """
        if not positive_only:
            return y_true, y_pred
        else:
            if one_dim:
                y_true_positive = y_true[y_pred > 0]
                y_pred_positive = y_pred[y_pred > 0]
                return y_true_positive, y_pred_positive
            else:
                y_true_positive = y_true[np.all(y_pred > 0, axis=1)]
                y_pred_positive = y_pred[np.all(y_pred > 0, axis=1)]
                return y_true_positive, y_pred_positive

    def __get_used_data(self, clean, y_true, y_pred, y_true_clean, y_pred_clean, one_dim):
        if clean:
            return y_true_clean, y_pred_clean, one_dim
        else:
            return y_true, y_pred, one_dim

    def __multi_output_result(self, result=None, multi_output=None, decimal=None):
        if isinstance(multi_output, (tuple, list, set, np.ndarray)):
            weights = np.array(multi_output)
            if self.y_true.ndim != len(weights):
                print("Permetrics Error! Multi-output weights has different length with y_true")
                exit(0)
            return np.round(np.dot(result, multi_output), decimal)
        else:  # Default: raw_values
            return np.round(result, decimal)

    def get_clean_data(self, y_true=None, y_pred=None, clean=False):
        """
        Get the cleaned data, the data pass to function will have higher priority than data pass to class object

        Args:
            y_true (tuple, list, np.ndarray):
            y_pred (tuple, list, np.ndarray):
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred)

        Returns:
            y_true_used: y_true used in computation process.
            y_pred_used: y_pred used in computation process
            dim: number of dimension in y_true
        """
        if y_true is not None and y_pred is not None:
            self.y_true_original, self.y_pred_original = deepcopy(y_true), deepcopy(y_pred)
            self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim = self.__clean_data(y_true, y_pred)
            return self.__get_used_data(clean, self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim)
        else:
            if self.y_true is not None and self.y_pred is not None:
                if self.already_clean:
                    return self.__get_used_data(clean, self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim)
                else:
                    self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim = self.__clean_data(y_true, y_pred)
                    self.already_clean = True
                    return self.__get_used_data(clean, self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim)
            else:
                print("Permetrics Error! You need to pass y_true and y_pred to object creation or function called.")
                exit(0)

    def get_preprocessed_data(self, y_true=None, y_pred=None, clean=False, decimal=None, positive_only=False):
        y_true, y_pred, one_dim = self.get_clean_data(y_true, y_pred, clean)
        y_true, y_pred = self.__positive_data(y_true, y_pred, one_dim, positive_only)
        decimal = self.decimal if decimal is None else decimal
        return y_true, y_pred, one_dim, decimal

    def explained_variance_score(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Explained Variance Score (EVS). Best possible score is 1.0, lower values are worse. Range = (-inf, 1.0]

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): EVS metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(1 - np.var(y_true - y_pred) / np.var(y_true), decimal)
        else:
            result = 1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def max_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Max Error (ME): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): ME metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.max(np.abs(y_true - y_pred)), decimal)
        else:
            result = np.max(np.abs(y_true - y_pred), axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def mean_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Mean Absolute Error (MAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MAE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.sum(np.abs(y_pred - y_true)) / len(y_true), decimal)
        else:
            result = np.sum(np.abs(y_pred - y_true), axis=0) / len(y_true)
            return self.__multi_output_result(result, multi_output, decimal)

    def mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Mean Squared Error (MSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.sum((y_pred - y_true) ** 2) / len(y_true), decimal)
        else:
            result = np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true)
            return self.__multi_output_result(result, multi_output, decimal)

    def root_mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Root Mean Squared Error (RMSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): RMSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.sqrt(np.sum((y_pred - y_true) ** 2) / len(y_true)), decimal)
        else:
            result = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true))
            return self.__multi_output_result(result, multi_output, decimal)

    def mean_squared_log_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=True, positive_only=True):
        """
        Mean Squared Log Error (MSLE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (float, int, np.ndarray): MSLE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.sum(np.log(y_true / y_pred) ** 2) / len(y_true), decimal)
        else:
            result = np.sum(np.log(y_true / y_pred) ** 2, axis=0) / len(y_true)
            return self.__multi_output_result(result, multi_output, decimal)

    def median_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Median Absolute Error (MedAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MedAE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.median(np.abs(y_true - y_pred)), decimal)
        else:
            result = np.median(np.abs(y_true - y_pred), axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def mean_relative_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=True, positive_only=False):
        """
        Mean Relative Error (MRE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MRE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.mean(np.divide(np.abs(y_true - y_pred), y_true)), decimal)
        else:
            result = np.mean(np.divide(np.abs(y_true - y_pred), y_true), axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=True, positive_only=False):
        """
        Mean Absolute Percentage Error (MAPE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MAPE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)), decimal)
        else:
            result = np.mean(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def symmetric_mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=True, positive_only=False):
        """
        Symmetric Mean Absolute Percentage Error (SMAPE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Link:
            + https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): SMAPE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))), decimal)
        else:
            result = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)), axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def mean_arctangent_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Mean Arctangent Absolute Percentage Error (MAAPE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Link:
            + https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MAAPE metric for single column or multiple columns (radian values)
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.mean(np.arctan(np.abs((y_true - y_pred)/y_true))), decimal)
        else:
            result = np.mean(np.arctan(np.abs((y_true - y_pred)/y_true)), axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def mean_absolute_scaled_error(self, y_true=None, y_pred=None, m=1, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Mean Absolute Scaled Error (MASE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Link:
            + https://en.wikipedia.org/wiki/Mean_absolute_scaled_error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            m (int): m = 1 for non-seasonal data, m > 1 for seasonal data
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MASE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[m:] - y_true[:-m])), decimal)
        else:
            result = np.mean(np.abs(y_true - y_pred), axis=0) / np.mean(np.abs(y_true[m:] - y_true[:-m]), axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def nash_sutcliffe_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Nash-Sutcliffe Efficiency (NSE): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]
        Link:
            + https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): NSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2), decimal)
        else:
            result = 1 - np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def willmott_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Willmott Index (WI): Best possible score is 1.0, bigger value is better. Range = [0, 1]
        Notes
        ~~~~~
            + Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            + https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): WI metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            m1 = np.mean(y_true)
            return np.round(1 - np.sum((y_pred - y_true) ** 2) / np.sum((np.abs(y_pred - m1) + np.abs(y_true - m1)) ** 2), decimal)
        else:
            m1 = np.mean(y_true, axis=0)
            result = 1 - np.sum((y_pred - y_true) ** 2, axis=0) / np.sum((np.abs(y_pred - m1) + np.abs(y_true - m1)) ** 2, axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def coefficient_of_determination(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Coefficient of Determination (R2): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]
        Notes
        ~~~~~
            + https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
            + This is not R^2 (or R*R), and should be denoted as R2, not like above scikit-learn website.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): R2 metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            return np.round(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2), decimal)
        else:
            result = 1 - np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
            return self.__multi_output_result(result, multi_output, decimal)

    def pearson_correlation_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Pearson’s Correlation Coefficient (PCC or R): Best possible score is 1.0, bigger value is better. Range = [-1, 1]
        Notes
        ~~~~~
            + Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            + Remember no absolute in the equations
            + https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): R metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_preprocessed_data(y_true, y_pred, clean, decimal, positive_only)
        if one_dim:
            m1, m2 = np.mean(y_true), np.mean(y_pred)
            return np.round(np.sum((y_true - m1) * (y_pred - m2)) / (np.sqrt(np.sum((y_true - np.m1) ** 2)) * np.sqrt(np.sum((y_pred - m2) ** 2))), decimal)
        else:
            m1, m2 = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
            numerator = np.sum((y_true - m1) * (y_pred - m2), axis=0)
            denominator = np.sqrt(np.sum((y_true - m1) ** 2, axis=0)) * np.sqrt(np.sum((y_pred - m2) ** 2, axis=0))
            return self.__multi_output_result(numerator / denominator, multi_output, decimal)

    def pearson_correlation_coefficient_square(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        (Pearson’s Correlation Index)^2 = R^2 = R2s (R square): Best possible score is 1.0, bigger value is better. Range = [0, 1]
        Notes
        ~~~~~
            + Do not misunderstand between R2s and R2 (Coefficient of Determination), they are different
            + Most of online tutorials (article, wikipedia,...) or even scikit-learn library are denoted the wrong R2s and R2.
            + R^2 = R2s = R squared should be (Pearson’s Correlation Index)^2
            + Meanwhile, R2 = Coefficient of Determination
            + https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): R2s metric for single column or multiple columns
        """
        result = self.pearson_correlation_coefficient(y_true, y_pred, multi_output, decimal, clean, positive_only)
        return np.round(result ** 2, decimal)

    def confidence_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, clean=False, positive_only=False):
        """
        Confidence Index (or Performance Index): CI (PI): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods
        Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            + > 0.85,          Excellent
            + 0.76-0.85,       Very good
            + 0.66-0.75,       Good
            + 0.61-0.65,       Satisfactory
            + 0.51-0.60,       Poor
            + 0.41-0.50,       Bad
            + ≤ 0.40,          Very bad

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive_only (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): CI (PI) metric for single column or multiple columns
        """
        r = self.pearson_correlation_coefficient(y_true, y_pred, multi_output, decimal, clean, positive_only)
        d = self.willmott_index(y_true, y_pred, multi_output, decimal, clean, positive_only)
        return np.round(r * d, decimal)

    def deviation_of_runoff_volume(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Deviation of Runoff Volume (DRV)
            https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            return round(sum(y_pred)/sum(y_true), decimal)
        else:
            temp = sum(y_pred, axis=0) / sum(y_true, axis=0)
            return self.__multi_output_result(temp, multi_output, decimal)

    def kling_gupta_efficiency(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Kling-Gupta Efficiency (KGE)
            https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        r = self.pearson_correlation_coefficient(clean, multi_output, decimal, y_true=y_true, y_pred=y_pred)
        if onedim:
            beta = mean(y_pred)/mean(y_true)
            gamma = (std(y_pred)/mean(y_pred))/(std(y_true)/mean(y_true))
            out = 1 - sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
            return round(out, decimal)
        else:
            beta = mean(y_pred, axis=0) / mean(y_true, axis=0)
            gamma = (std(y_pred, axis=0) / mean(y_pred, axis=0)) / (std(y_true, axis=0) / mean(y_true, axis=0))
            out = 1 - sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
            return self.__multi_output_result(out, multi_output, decimal)

    def gini_coefficient(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Gini coefficient (Gini)
            https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/gini.m
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            idx_sort = argsort(-y_pred)
            population_delta = 1.0 / len(y_true)
            accumulated_population_percentage_sum, accumulated_loss_percentage_sum, score = 0, 0, 0
            total_losses = sum(y_true)
            for i in range(0, len(y_true)):
                accumulated_loss_percentage_sum += y_true[idx_sort[i]] / total_losses
                accumulated_population_percentage_sum += population_delta
                score += accumulated_loss_percentage_sum - accumulated_population_percentage_sum
            score = score / len(y_true)
            return round(score, decimal)
        else:
            col = y_true.shape[1]
            idx_sort = argsort(-y_pred, axis=0)
            population_delta = 1.0 / len(y_true)
            accumulated_population_percentage_sum, accumulated_loss_percentage_sum, score = zeros(col), zeros(col), zeros(col)
            total_losses = sum(y_true, axis=0)
            for i in range(0, col):
                for j in range(0, len(y_true)):
                    accumulated_loss_percentage_sum[i] += y_true[idx_sort[j, i], i] / total_losses[i]
                    accumulated_population_percentage_sum[i] += population_delta
                    score[i] += accumulated_loss_percentage_sum[i] - accumulated_population_percentage_sum[i]
            score = score / len(y_true)
            return self.__multi_output_result(score, multi_output, decimal)

    def gini_coefficient_wiki(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Gini coefficient (Gini)
            https://en.wikipedia.org/wiki/Gini_coefficient
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            y = concatenate((y_true, y_pred), axis=0)
            score = 0
            for i in range(0, len(y)):
                for j in range(0, len(y)):
                    score += abs(y[i] - y[j])
            y_mean = mean(y)
            score = score / (2*len(y)**2 * y_mean)
            return round(score, decimal)
        else:
            y = concatenate((y_true, y_pred), axis=0)
            col = y.shape[1]
            d = len(y)
            score = zeros(col)
            for k in range(0, col):
                for i in range(0, d):
                    for j in range(0, d):
                        score[k] += abs(y[i, k] - y[j, k])
            y_mean = mean(y, axis=0)
            score = score / (2 * len(y) ** 2 * y_mean)
            return self.__multi_output_result(score, multi_output, decimal)

    def prediction_of_change_in_direction(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Prediction of change in direction
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            d = diff(y_true)
            dp = diff(y_pred)
            return round(mean(sign(d) == sign(dp)), decimal)
        else:
            d = diff(y_true, axis=0)
            dp = diff(y_pred, axis=0)
            score = mean(sign(d) == sign(dp), axis=0)
            return self.__multi_output_result(score, multi_output, decimal)

    def entropy(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Entropy Loss function
            https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            score = -sum(y_true * log(y_pred.clip(self.EPSILON, None)))
            return round(score, decimal)
        else:
            score = -sum(y_true * log(y_pred.clip(self.EPSILON, None)), axis=0)
            return self.__multi_output_result(score, multi_output, decimal)

    def cross_entropy(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            f_true, intervals = histogram(y_true, bins=len(unique(y_true)) - 1)
            intervals[0] = min([min(y_true), min(y_pred)])
            intervals[-1] = max([max(y_true), max(y_pred)])
            f_true = f_true / len(f_true)
            f_pred = histogram(y_pred, bins=intervals)[0] / len(y_pred)
            value = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_pred)
            return round(value, decimal)
        else:
            score = []
            for i in range(y_true.shape[1]):
                f_true, intervals = histogram(y_true[:,i], bins=len(unique(y_true[:,i])) - 1)
                intervals[0] = min([min(y_true[:,i]), min(y_pred[:,i])])
                intervals[-1] = max([max(y_true[:,i]), max(y_pred[:,i])])
                f_true = f_true / len(f_true)
                f_pred = histogram(y_pred[:,i], bins=intervals)[0] / len(y_pred[:,i])
                score.append(self.entropy(clean, "raw_values", decimal, y_true=f_true, y_pred=f_pred))
            return self.__multi_output_result(score, multi_output, decimal)

    def kullback_leibler_divergence(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Kullback-Leibler Divergence
        https://machinelearningmastery.com/divergence-between-probability-distributions/
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            f_true, intervals = histogram(y_true, bins=len(unique(y_true)) - 1)
            intervals[0] = min([min(y_true), min(y_pred)])
            intervals[-1] = max([max(y_true), max(y_pred)])
            f_true = f_true / len(f_true)
            f_pred = histogram(y_pred, bins=intervals)[0] / len(y_pred)
            score = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_pred) - \
                    self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
            return round(score, decimal)
        else:
            score = []
            for i in range(y_true.shape[1]):
                f_true, intervals = histogram(y_true[:, i], bins=len(unique(y_true[:, i])) - 1)
                intervals[0] = min([min(y_true[:, i]), min(y_pred[:, i])])
                intervals[-1] = max([max(y_true[:, i]), max(y_pred[:, i])])
                f_true = f_true / len(f_true)
                f_pred = histogram(y_pred[:, i], bins=intervals)[0] / len(y_pred[:, i])
                temp1 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_pred)
                temp2 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
                score.append(temp1 - temp2)
            return self.__multi_output_result(score, multi_output, decimal)

    def jensen_shannon_divergence(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Jensen-Shannon Divergence
        https://machinelearningmastery.com/divergence-between-probability-distributions/
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            f_true, intervals = histogram(y_true, bins=len(unique(y_true)) - 1)
            intervals[0] = min([min(y_true), min(y_pred)])
            intervals[-1] = max([max(y_true), max(y_pred)])
            f_true = f_true / len(f_true)
            f_pred = histogram(y_pred, bins=intervals)[0] / len(y_pred)
            m = 0.5 * (f_true + f_pred)
            temp1 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=m) - \
                    self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
            temp2 = self.entropy(clean, None, decimal, y_true=f_pred, y_pred=m) - \
                    self.entropy(clean, None, decimal, y_true=f_pred, y_pred=f_pred)
            return round(0.5 * temp1 + 0.5 * temp2, decimal)
        else:
            score = []
            for i in range(y_true.shape[1]):
                f_true, intervals = histogram(y_true[:, i], bins=len(unique(y_true[:, i])) - 1)
                intervals[0] = min([min(y_true[:, i]), min(y_pred[:, i])])
                intervals[-1] = max([max(y_true[:, i]), max(y_pred[:, i])])
                f_true = f_true / len(f_true)
                f_pred = histogram(y_pred[:, i], bins=intervals)[0] / len(y_pred[:, i])
                m = 0.5 * (f_true + f_pred)
                temp1 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=m) - \
                        self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
                temp2 = self.entropy(clean, None, decimal, y_true=f_pred, y_pred=m) - \
                        self.entropy(clean, None, decimal, y_true=f_pred, y_pred=f_pred)
                score.append(0.5 * temp1 + 0.5 * temp2)
            return self.__multi_output_result(score, multi_output, decimal)

    def variance_accounted_for(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Variance Accounted For between 2 signals
        https://www.dcsc.tudelft.nl/~jwvanwingerden/lti/doc/html/vaf.html
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            vaf = (1 - (y_true - y_pred).var()/y_true.var()) * 100
            return round(vaf, decimal)
        else:
            vaf = (1 - (y_true - y_pred).var(axis=0) / y_true.var(axis=0)) * 100
            return self.__multi_output_result(vaf, multi_output, decimal)

    def relative_absolute_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Relative Absolute Error
        https://stackoverflow.com/questions/59499222/how-to-make-a-function-of-mae-and-rae-without-using-librarymetrics
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            mean_true = mean(y_true)
            rae = sum(abs(y_true - y_pred)) / sum(abs(y_true - mean_true))
            return round(rae, decimal)
        else:
            mean_true = mean(y_true, axis=0)
            rae = sum(abs(y_true - y_pred), axis=0) / sum(abs(y_true - mean_true), axis=0)
            return self.__multi_output_result(rae, multi_output, decimal)

    def a10_index(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.9, div <=1.1), 1, 0)
            return round(mean(div), decimal)
        else:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.9, div <= 1.1), 1, 0)
            return self.__multi_output_result(mean(div, axis=0), multi_output, decimal)

    def a20_index(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return round(mean(div), decimal)
        else:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return self.__multi_output_result(mean(div, axis=0), multi_output, decimal)

    def normalized_root_mean_square_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
        Normalized Root Mean Square Error
        https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        rmse = self.root_mean_squared_error(clean, multi_output, decimal, y_true=y_true, y_pred=y_pred)
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = 0
        if onedim:
            if model == 0:
                dif = max(y_true) - min(y_true)
                return round(rmse/dif, decimal)
            elif model == 1:
                mean_pred = mean(y_pred)
                return round(rmse/mean_pred, decimal)
            elif model == 2:
                std_pred = y_pred.std()
                return round(rmse/std_pred, decimal)
            else:
                value = sqrt(sum(log((y_pred+1) / (y_true+1))**2)/len(y_true))
                return round(value, decimal)
        else:
            if model == 0:
                dif = max(y_true, axis=0) - min(y_true, axis=0)
                value = rmse / dif
            elif model == 1:
                mean_pred = mean(y_pred, axis=0)
                value = rmse / mean_pred
            elif model == 2:
                std_pred = y_pred.std(axis=0)
                value = rmse / std_pred
            else:
                value = sqrt(sum(log((y_pred + 1) / (y_true + 1)) ** 2, axis=0) / len(y_true))
            return self.__multi_output_result(value, multi_output, decimal)

    def residual_standard_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
        Residual Standard Error
        https://www.statology.org/residual-standard-error-r/
        """
        y_true, y_pred, onedim = self.get_clean_data(clean, kwargs)
        if onedim:
            x = y_pred
            y = y_true / y_pred
            up = (sum((x - mean(x)) * (y - mean(y)))) ** 2
            down = sum((x - mean(x)) ** 2) * sum((y - mean(y)) ** 2)
            return round(up/down, decimal)
        else:
            x = y_pred
            y = y_true / y_pred
            up = (sum((x - mean(x, axis=0)) * (y - mean(y, axis=0)), axis=0)) ** 2
            down = sum((x - mean(x, axis=0)) ** 2, axis=0) * sum((y - mean(y, axis=0)) ** 2, axis=0)
            return round(up / down, decimal)

    def get_metric_by_name(self, func_name:str, paras=None) -> dict:
        """
        Parameters
        ----------
        func_name : str. For example: "RMSE"
        paras : dict. For example:
            Default: Don't specify it. leave it there
            Else: It has to be a dictionary such as {"decimal": 3, "multi_output": "raw_values", }

        Returns
        -------
        dict: { "RMSE": 0.2 }
        """
        temp = {}
        obj = getattr(self, func_name)
        if paras is None:
            temp[func_name] = obj()
        else:
            temp[func_name] = obj(**paras)
        return temp

    def get_metrics_by_list_names(self, list_func_names:list, list_paras=None) -> dict:
        """
        Parameters
        ----------
        func_names : list. For example: ["RMSE", "MAE", "MAPE"]
        paras : list. For example: [ {"decimal": 5, }, None, { "decimal": 4, "multi_output": "raw_values" } }       # List of dict

        Returns
        -------
        dict. For example: { "RMSE": 0.25, "MAE": 0.7, "MAPE": 0.15 }
        """
        temp = {}
        for idx, func_name in enumerate(list_func_names):
            obj = getattr(self, func_name)
            if list_paras is None:
                temp[func_name] = obj()
            else:
                if len(list_func_names) != len(list_paras):
                    print("Failed! Different length between list of functions and list of parameters")
                    exit(0)
                if list_paras[idx] is None:
                    temp[func_name] = obj()
                else:
                    temp[func_name] = obj(**list_paras[idx])
        return temp

    def get_metrics_by_dict(self, metrics_dict:dict) -> dict:
        """
        Parameters
        ----------
        metrics_dict : dict inside dict, for examples:
            {
                "RMSE": { "multi_output": multi_output, "decimal": 4 }
                "MAE": { "clean": True, "multi_output": multi_output, "decimal": 6 }
            }
        Returns
        -------
            A dict: For example
                { "RMSE": 0.3524, "MAE": 0.445263 }
        """
        temp = {}
        for func_name, paras_dict in metrics_dict.items():
            obj = getattr(self, func_name)
            if paras_dict is None:
                temp[func_name] = obj()
            else:
                temp[func_name] = obj(**paras_dict)     # Unpacking a dictionary and passing it to function
        return temp

    EVS = evs = explained_variance_score
    ME = me = max_error
    MAE = mae = mean_absolute_error
    MSE = mse = mean_squared_error
    RMSE = rmse = root_mean_squared_error
    MSLE = msle = mean_squared_log_error
    MedAE = medae = median_absolute_error
    MRE = mre = mean_relative_error
    MAPE = mape = mean_absolute_percentage_error
    SMAPE = smape = symmetric_mean_absolute_percentage_error
    MAAPE = maape = mean_arctangent_absolute_percentage_error
    MASE = mase = mean_absolute_scaled_error
    NSE = nse = nash_sutcliffe_efficiency
    WI = wi = willmott_index
    R = r = pearson_correlation_coefficient
    R2s = r2s = pearson_correlation_coefficient_square
    CI = ci = confidence_index
    R2 = r2 = coefficient_of_determination
    DRV = drv = deviation_of_runoff_volume
    KGE = kge = kling_gupta_efficiency
    GINI = gini = gini_coefficient
    GINI_WIKI = gini_wiki = gini_coefficient_wiki
    PCD = pcd = prediction_of_change_in_direction
    E = e = entropy
    CE = ce = cross_entropy
    KLD = kld = kullback_leibler_divergence
    JSD = jsd = jensen_shannon_divergence
    VAF = vaf = variance_accounted_for
    RAE = rae = relative_absolute_error
    A10 = a10 = a10_index
    A20 = a20 = a20_index
    NRMSE = nrmse = normalized_root_mean_square_error
    RSE = rse = residual_standard_error