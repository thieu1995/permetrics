# !/usr/bin/env python
# Created by "Thieu" at 18:07, 18/07/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from permetrics.evaluator import Evaluator
from permetrics.utils import regressor_util as ru
from permetrics.utils import data_util as du
import numpy as np


class RegressionMetric(Evaluator):
    """
    Defines a RegressionMetric class that hold all regression metrics (for both regression and time-series problems)

    + An extension of scikit-learn metrics section, with the addition of many more regression metrics.
    + https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    + Some methods in scikit-learn can't generate the multi-output metrics, we re-implement all of them and allow multi-output metrics
    + Therefore, we support calculate the multi-output metrics for all methods

    Parameters
    ----------
    y_true: tuple, list, np.ndarray, default = None
        The ground truth values.

    y_pred: tuple, list, np.ndarray, default = None
        The prediction values.

    decimal: int, default = 5
        The number of fractional parts after the decimal point
    """

    SUPPORT = {
        "EVS": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "ME": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MBE": {"type": "unknown", "range": "(-inf, +inf)", "best": "0"},
        "MAE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MSE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "RMSE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MSLE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MedAE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MRE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MRB": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MPE": {"type": "unknown", "range": "(-inf, +inf)", "best": "0"},
        "MAPE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "SMAPE": {"type": "min", "range": "[0, 1]", "best": "0"},
        "MAAPE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MASE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "NSE": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "NNSE": {"type": "max", "range": "[0, 1]", "best": "1"},
        "WI": {"type": "max", "range": "[0, 1]", "best": "1"},
        "R": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "PCC": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "AR": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "APCC": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "R2S": {"type": "max", "range": "[0, 1]", "best": "1"},
        "RSQ": {"type": "max", "range": "[0, 1]", "best": "1"},
        "R2": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "COD": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "AR2": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "ACOD": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "CI": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "DRV": {"type": "min", "range": "[1, +inf)", "best": "1"},
        "KGE": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "GINI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "GINI_WIKI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "PCD": {"type": "max", "range": "[0, 1]", "best": "1"},
        "CE": {"type": "unknown", "range": "(-inf, 0]", "best": "unknown"},
        "KLD": {"type": "unknown", "range": "(-inf, +inf)", "best": "0"},
        "JSD": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "VAF": {"type": "max", "range": "(-inf, 100%)", "best": "100"},
        "RAE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "A10": {"type": "max", "range": "[0, 1]", "best": "1"},
        "A20": {"type": "max", "range": "[0, 1]", "best": "1"},
        "A30": {"type": "max", "range": "[0, 1]", "best": "1"},
        "NRMSE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "RSE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "RE": {"type": "unknown", "range": "(-inf, +inf)", "best": "0"},
        "RB": {"type": "unknown", "range": "(-inf, +inf)", "best": "0"},
        "AE": {"type": "unknown", "range": "(-inf, +inf)", "best": "0"},
        "SE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "SLE": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "COV": {"type": "max", "range": "(-inf, +inf)", "best": "no best"},
        "COR": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "EC": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "OI": {"type": "max", "range": "(-inf, 1]", "best": "1"},
        "CRM": {"type": "min", "range": "(-inf, +inf)", "best": "0"},
    }

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        super().__init__(y_true, y_pred, decimal, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.one_dim = False

    @staticmethod
    def get_support(name=None, verbose=True):
        if name == "all":
            if verbose:
                for key, value in RegressionMetric.SUPPORT.items():
                    print(f"Metric {key} : {value}")
            return RegressionMetric.SUPPORT
        if name not in list(RegressionMetric.SUPPORT.keys()):
            raise ValueError(f"RegressionMetric doesn't support metric named: {name}")
        else:
            if verbose:
                print(f"Metric {name}: {RegressionMetric.SUPPORT[name]}")
            return RegressionMetric.SUPPORT[name]

    def get_processed_data(self, y_true=None, y_pred=None, decimal=None, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred)
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            one_dim: is y_true has 1 dimensions or not
            decimal: The number of fractional parts after the decimal point
        """
        decimal = self.decimal if decimal is None else decimal
        if (y_true is not None) and (y_pred is not None):
            y_true, y_pred = du.format_regression_data_type(y_true, y_pred)
            y_true, y_pred, one_dim = du.format_regression_data(y_true, y_pred)
        else:
            if (self.y_true is not None) and (self.y_pred is not None):
                y_true, y_pred = du.format_regression_data_type(self.y_true, self.y_pred)
                y_true, y_pred, one_dim = du.format_regression_data(y_true, y_pred)
            else:
                raise ValueError("y_true or y_pred is None. You need to pass y_true and y_pred to object creation or function called.")
        return y_true, y_pred, one_dim, decimal

    def explained_variance_score(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Explained Variance Score (EVS). Best possible score is 1.0, greater value is better. Range = (-inf, 1.0]

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in denominator (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): EVS metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(1 - np.var(y_true - y_pred) / np.var(y_true), decimal)
        else:
            result = 1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def max_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Max Error (ME): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): ME metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.max(np.abs(y_true - y_pred)), decimal)
        else:
            result = np.max(np.abs(y_true - y_pred), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_bias_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Mean Bias Error (MBE): Best possible score is 0.0. Range = (-inf, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MBE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(y_pred - y_true), decimal)
        else:
            result = np.mean(y_pred - y_true, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Mean Absolute Error (MAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MAE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(np.abs(y_pred - y_true)) / len(y_true), decimal)
        else:
            result = np.sum(np.abs(y_pred - y_true), axis=0) / len(y_true)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Mean Squared Error (MSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        score = ru.calculate_mse(y_true, y_pred, one_dim)
        return np.round(score, decimal) if one_dim else self.get_multi_output_result(score, multi_output, decimal)

    def root_mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Root Mean Squared Error (RMSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): RMSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        score = np.sqrt(ru.calculate_mse(y_true, y_pred, one_dim))
        return np.round(score, decimal) if one_dim else self.get_multi_output_result(score, multi_output, decimal)

    def mean_squared_log_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, **kwargs):
        """
        Mean Squared Log Error (MSLE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Link: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error-(msle)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (float, int, np.ndarray): MSLE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(np.log((y_true + 1) / (y_pred+1)) ** 2) / len(y_true), decimal)
        else:
            result = np.sum(np.log((y_true + 1) / (y_pred + 1)) ** 2, axis=0) / len(y_true)
            return self.get_multi_output_result(result, multi_output, decimal)

    def median_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Median Absolute Error (MedAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MedAE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.median(np.abs(y_true - y_pred)), decimal)
        else:
            result = np.median(np.abs(y_true - y_pred), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_relative_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False, **kwargs):
        """
        Mean Relative Error (MRE) - Mean Relative Bias (MRB): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MRE (MRB) metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 0)
        else:
            y_true[y_true == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs((y_pred - y_true) / y_true)), decimal)
        else:
            result = np.mean(np.abs((y_pred - y_true) / y_true), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False, **kwargs):
        """
        Mean Percentage Error (MPE): Best possible score is 0.0. Range = (-inf, +inf)
        Link: https://www.dataquest.io/blog/understanding-regression-error-metrics/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MPE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 0)
        else:
            y_true[y_true == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean((y_true - y_pred) / y_true), decimal)
        else:
            result = np.mean((y_true - y_pred) / y_true, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False, **kwargs):
        """
        Mean Absolute Percentage Error (MAPE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MAPE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 0)
        else:
            y_true[y_true == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)), decimal)
        else:
            result = np.mean(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def symmetric_mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values",
                                                 decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Symmetric Mean Absolute Percentage Error (SMAPE): Best possible score is 0.0, smaller value is better. Range = [0, 1]
        If you want percentage then multiply with 100%

        Link: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): SMAPE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))), decimal)
        else:
            result = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_arctangent_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None,
                                                  non_zero=False, positive=False, **kwargs):
        """
        Mean Arctangent Absolute Percentage Error (MAAPE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Link: https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MAAPE metric for single column or multiple columns (radian values)
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.arctan(np.abs((y_true - y_pred)/y_true))), decimal)
        else:
            result = np.mean(np.arctan(np.abs((y_true - y_pred)/y_true)), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_absolute_scaled_error(self, y_true=None, y_pred=None, m=1, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Mean Absolute Scaled Error (MASE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Link: https://en.wikipedia.org/wiki/Mean_absolute_scaled_error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            m (int): m = 1 for non-seasonal data, m > 1 for seasonal data
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MASE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[m:] - y_true[:-m])), decimal)
        else:
            result = np.mean(np.abs(y_true - y_pred), axis=0) / np.mean(np.abs(y_true[m:] - y_true[:-m]), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def nash_sutcliffe_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Nash-Sutcliffe Efficiency (NSE): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]

        Link: https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): NSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        nse = ru.calculate_nse(y_true, y_pred, one_dim)
        return np.round(nse, decimal) if one_dim else self.get_multi_output_result(nse, multi_output, decimal)

    def normalized_nash_sutcliffe_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None,
                                             non_zero=False, positive=False, **kwargs):
        """
        Normalize Nash-Sutcliffe Efficiency (NNSE): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Link: https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): NSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        nse = ru.calculate_nse(y_true, y_pred, one_dim)
        nnse = 1 / (2 - nse)
        return np.round(nnse, decimal) if one_dim else self.get_multi_output_result(nnse, multi_output, decimal)

    def willmott_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
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
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): WI metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        wi = ru.calculate_wi(y_true, y_pred, one_dim)
        return np.round(wi, decimal) if one_dim else self.get_multi_output_result(wi, multi_output, decimal)

    def coefficient_of_determination(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Coefficient of Determination (COD/R2): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]

        Notes
        ~~~~~
            + https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
            + Scikit-learn and other websites denoted COD as R^2 (or R squared), it leads to the misunderstanding of R^2 in which R is PCC.
            + We should denote it as COD or R2 only.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): R2 metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2), decimal)
        else:
            result = 1 - np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def adjusted_coefficient_of_determination(self, y_true=None, y_pred=None, X_shape=None, multi_output="raw_values", decimal=None,
                                              non_zero=False, positive=False, **kwargs):
        """
        Adjusted Coefficient of Determination (ACOD/AR2): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]

        Notes
        ~~~~~
            + https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
            + Scikit-learn and other websites denoted COD as R^2 (or R squared), it leads to the misunderstanding of R^2 in which R is PCC.
            + We should denote it as COD or R2 only.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            X_shape (tuple, list, np.ndarray): The shape of X_train dataset
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): AR2 metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if X_shape is None:
            raise ValueError("You need to pass the shape of X_train dataset to calculate Adjusted R2.")
        if len(X_shape) != 2 or X_shape[0] < 4 or X_shape[1] < 1:
            raise ValueError("You need to pass the real shape of X_train dataset to calculate Adjusted R2.")
        dft = X_shape[0] - 1.0
        dfe = X_shape[0] - X_shape[1] - 1.0
        df_final = dft/dfe
        if one_dim:
            return np.round(1 - df_final * np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2), decimal)
        else:
            result = 1 - df_final * np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def pearson_correlation_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
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
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): R metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        pcc = ru.calculate_pcc(y_true, y_pred, one_dim)
        return np.round(pcc, decimal) if one_dim else self.get_multi_output_result(pcc, multi_output, decimal)

    def absolute_pearson_correlation_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None,
                                                 non_zero=False, positive=False, **kwargs):
        """
        Absolute Pearson’s Correlation Coefficient (APCC or AR): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): AR metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        pcc = ru.calculate_absolute_pcc(y_true, y_pred, one_dim)
        return np.round(pcc, decimal) if one_dim else self.get_multi_output_result(pcc, multi_output, decimal)

    def pearson_correlation_coefficient_square(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None,
                                               non_zero=False, positive=False, **kwargs):
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
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): R2s metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        pcc = ru.calculate_pcc(y_true, y_pred, one_dim)
        return np.round(pcc**2, decimal) if one_dim else self.get_multi_output_result(pcc**2, multi_output, decimal)

    def confidence_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Confidence Index (or Performance Index): CI (PI): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]

        Notes
        ~~~~~
        - Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
        - > 0.85,          Excellent
        - 0.76-0.85,       Very good
        - 0.66-0.75,       Good
        - 0.61-0.65,       Satisfactory
        - 0.51-0.60,       Poor
        - 0.41-0.50,       Bad
        - < 0.40,          Very bad

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): CI (PI) metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        r = ru.calculate_pcc(y_true, y_pred, one_dim)
        d = ru.calculate_wi(y_true, y_pred, one_dim)
        return np.round(r*d, decimal) if one_dim else self.get_multi_output_result(r*d, multi_output, decimal)

    def deviation_of_runoff_volume(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Deviation of Runoff Volume (DRV): Best possible score is 1.0, smaller value is better. Range = [1, +inf)
        Link: https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): DRV metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(y_pred) / np.sum(y_true), decimal)
        else:
            result = np.sum(y_pred, axis=0) / np.sum(y_true, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def kling_gupta_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Kling-Gupta Efficiency (KGE): Best possible score is 1, bigger value is better. Range = (-inf, 1]
        Link: https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): KGE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            m1, m2 = np.mean(y_true), np.mean(y_pred)
            r = np.sum((y_true - m1) * (y_pred - m2)) / (np.sqrt(np.sum((y_true - m1) ** 2)) * np.sqrt(np.sum((y_pred - m2) ** 2)))
            beta = m2 / m1
            gamma = (np.std(y_pred) / m2) / (np.std(y_true) / m1)
            return np.round(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2), decimal)
        else:
            m1, m2 = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
            num_r = np.sum((y_true - m1) * (y_pred - m2), axis=0)
            den_r = np.sqrt(np.sum((y_true - m1) ** 2, axis=0)) * np.sqrt(np.sum((y_pred - m2) ** 2, axis=0))
            r = num_r / den_r
            beta = m2 / m1
            gamma = (np.std(y_pred, axis=0) / m2) / (np.std(y_true, axis=0) / m1)
            result = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
            return self.get_multi_output_result(result, multi_output, decimal)

    def prediction_of_change_in_direction(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Prediction of Change in Direction (PCD): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): PCD metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            d = np.diff(y_true)
            dp = np.diff(y_pred)
            return np.round(np.mean(np.sign(d) == np.sign(dp)), decimal)
        else:
            d = np.diff(y_true, axis=0)
            dp = np.diff(y_pred, axis=0)
            result = np.mean(np.sign(d) == np.sign(dp), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def cross_entropy(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=True, **kwargs):
        """
        Cross Entropy (CE) or Entropy (E): Range = (-inf, 0]. Can't give comment about this one

        Notes
        ~~~~~
            + Greater value of Entropy, the greater the uncertainty for probability distribution and smaller the value the less the uncertainty
            + https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (float, int, np.ndarray): CE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        entropy = ru.calculate_entropy(y_true, y_pred, one_dim)
        return np.round(entropy, decimal) if one_dim else self.get_multi_output_result(entropy, multi_output, decimal)

    def kullback_leibler_divergence(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, **kwargs):
        """
        Kullback-Leibler Divergence (KLD): Best possible score is 0.0 . Range = (-inf, +inf)
        Link: https://machinelearningmastery.com/divergence-between-probability-distributions/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (float, int, np.ndarray): KLD metric (bits) for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(y_true * np.log2(y_true / y_pred)), decimal)
        else:
            score = np.sum(y_true * np.log2(y_true / y_pred), axis=0)
            return self.get_multi_output_result(score, multi_output, decimal)

    def jensen_shannon_divergence(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=True, **kwargs):
        """
        Jensen-Shannon Divergence (JSD): Best possible score is 0.0 (identical), smaller value is better . Range = [0, +inf)
        Link: https://machinelearningmastery.com/divergence-between-probability-distributions/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (float, int, np.ndarray): JSD metric (bits) for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            m = 0.5 * (y_true + y_pred)
            score = 0.5 * np.sum(y_true * np.log2(y_true / m)) + 0.5 * np.sum(y_pred * np.log2(y_pred / m))
            return np.round(score, decimal)
        else:
            m = 0.5 * (y_true + y_pred)
            score = 0.5 * np.sum(y_true * np.log2(y_true / m), axis=0) + 0.5 * np.sum(y_pred * np.log2(y_pred / m), axis=0)
            return self.get_multi_output_result(score, multi_output, decimal)

    def variance_accounted_for(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Variance Accounted For between 2 signals (VAF): Best possible score is 100% (identical signal), bigger value is better. Range = (-inf, 100%]

        Link: https://www.dcsc.tudelft.nl/~jwvanwingerden/lti/doc/html/vaf.html

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): VAF metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 0)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)

        if one_dim:
            return np.round((1 - np.var(y_true - y_pred) / np.var(y_true)) * 100, decimal)
        else:
            result = (1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)) * 100
            return self.get_multi_output_result(result, multi_output, decimal)

    def relative_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Relative Absolute Error (RAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Notes
        ~~~~~
            + https://stackoverflow.com/questions/59499222/how-to-make-a-function-of-mae-and-rae-without-using-librarymetrics
            + https://www.statisticshowto.com/relative-absolute-error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): RAE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            numerator = np.power(np.sum((y_pred - y_true)**2), 1/2)
            denominator = np.power(np.sum(y_true**2), 1/2)
            return np.round(numerator/denominator, decimal)
        else:
            numerator = np.power(np.sum((y_pred - y_true) ** 2, axis=0), 1 / 2)
            denominator = np.power(np.sum(y_true ** 2, axis=0), 1 / 2)
            return self.get_multi_output_result(numerator/denominator, multi_output, decimal)

    def a10_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False, **kwargs):
        """
        A10 index (A10): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Notes
        ~~~~~
            + a10-index is engineering index for evaluating artificial intelligence models by showing the number of samples
            + that fit the prediction values with a deviation of ±10% compared to experimental values
            + https://www.mdpi.com/2076-3417/9/18/3715/htm

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): A10 metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)

        if one_dim:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.9, div <= 1.1), 1, 0)
            return np.round(np.mean(div), decimal)
        else:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.9, div <= 1.1), 1, 0)
            return self.get_multi_output_result(np.mean(div, axis=0), multi_output, decimal)

    def a20_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False, **kwargs):
        """
        A20 index (A20): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Notes
        ~~~~~
            + a20-index evaluated metric by showing the number of samples that fit the prediction values with a deviation of ±20% compared to experimental values
            + https://www.mdpi.com/2076-3417/9/18/3715/htm

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): A20 metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return np.round(np.mean(div), decimal)
        else:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return self.get_multi_output_result(np.mean(div, axis=0), multi_output, decimal)

    def a30_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False, **kwargs):
        """
        A30 index (A30): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Note: a30-index evaluated metric by showing the number of samples that fit the prediction values with a deviation of ±30% compared to experimental values

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): A30 metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.7, div <= 1.3), 1, 0)
            return np.round(np.mean(div), decimal)
        else:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.7, div <= 1.3), 1, 0)
            return self.get_multi_output_result(np.mean(div, axis=0), multi_output, decimal)

    def normalized_root_mean_square_error(self, y_true=None, y_pred=None, model=0, multi_output="raw_values", decimal=None,
                                          non_zero=False, positive=False, **kwargs):
        """
        Normalized Root Mean Square Error (NRMSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Link: https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            model (int): Normalize RMSE by different ways, (Optional, default = 0, valid values = [0, 1, 2, 3]
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): NRMSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            if model == 1:
                result = rmse / np.mean(y_pred)
            elif model == 2:
                result = rmse / (np.max(y_true) - np.min(y_true))
            elif model == 3:
                result = np.sqrt(np.sum(np.log((y_pred + 1) / (y_true + 1)) ** 2) / len(y_true))
            else:
                result = rmse / y_pred.std()
            return np.round(result, decimal)
        else:
            rmse = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true))
            if model == 1:
                result = rmse / np.mean(y_pred, axis=0)
            elif model == 2:
                result = rmse / (np.max(y_true, axis=0) - np.min(y_true, axis=0))
            elif model == 3:
                result = np.sqrt(np.sum(np.log((y_pred + 1) / (y_true + 1)) ** 2, axis=0) / len(y_true))
            else:
                result = rmse / y_pred.std(axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def residual_standard_error(self, y_true=None, y_pred=None, n_paras=None, multi_output="raw_values", decimal=None,
                                non_zero=False, positive=False, **kwargs):
        """
        Residual Standard Error (RSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Links:
            + https://www.statology.org/residual-standard-error-r/
            + https://machinelearningmastery.com/degrees-of-freedom-in-machine-learning/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            n_paras (int): The number of model's parameters
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): RSE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if n_paras is None:
            n_paras = len(y_true) / 2
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            ss_residuals = np.sum((y_true - y_pred)**2)
            df_residuals = len(y_true) - n_paras - 1
            return np.round(np.sqrt(ss_residuals / df_residuals), decimal)
        else:
            ss_residuals = np.sum((y_true - y_pred) ** 2, axis=0)
            df_residuals = len(y_true) - n_paras - 1
            score = np.sqrt(ss_residuals / df_residuals)
            return self.get_multi_output_result(score, multi_output, decimal)

    def covariance(self, y_true=None, y_pred=None, sample=False, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Covariance (COV): There is no best value, bigger value is better. Range = [-inf, +inf)
            + is a measure of the relationship between two random variables
            + evaluates how much – to what extent – the variables change together
            + does not assess the dependency between variables
            + Positive covariance: Indicates that two variables tend to move in the same direction.
            + Negative covariance: Reveals that two variables tend to move in inverse directions.

        Links:
            + https://corporatefinanceinstitute.com/resources/data-science/covariance/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            sample (bool): sample covariance or population covariance. See the website above for more details
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): COV metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        denominator = len(y_true) - 1 if sample else len(y_true)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            s1 = y_true - np.mean(y_true)
            s2 = y_pred - np.mean(y_pred)
            return np.round(np.dot(s1, s2) / denominator, decimal)
        else:
            s1 = y_true - np.mean(y_true, axis=0)
            s2 = y_pred - np.mean(y_pred, axis=0)
            return self.get_multi_output_result(np.sum(s1 * s2, axis=0) / denominator, multi_output, decimal)

    def correlation(self, y_true=None, y_pred=None, sample=False, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Correlation (COR): Best possible value = 1, bigger value is better. Range = [-1, +1]
            + measures the strength of the relationship between variables
            + is the scaled measure of covariance. It is dimensionless.
            + the correlation coefficient is always a pure value and not measured in any units.

        Links:
            + https://corporatefinanceinstitute.com/resources/data-science/covariance/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            sample (bool): sample covariance or population covariance. See the website above for more details
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): COR metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        denominator = len(y_true) - 1 if sample else len(y_true)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            s1 = y_true - np.mean(y_true)
            s2 = y_pred - np.mean(y_pred)
            return np.round((np.dot(s1, s2) / denominator) / (np.std(y_true) * np.std(y_pred)), decimal)
        else:
            s1 = y_true - np.mean(y_true, axis=0)
            s2 = y_pred - np.mean(y_pred, axis=0)
            cov = np.sum(s1 * s2, axis=0) / denominator
            den = np.std(y_true, axis=0) * np.std(y_pred, axis=0)
            return self.get_multi_output_result(cov / den, multi_output, decimal)

    def efficiency_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Efficiency Coefficient (EC): Best possible value = 1, bigger value is better. Range = [-inf, +1]

        Links:
            + https://doi.org/10.1016/j.solener.2019.01.037

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): EC metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        score = ru.calculate_ec(y_true, y_pred, one_dim)
        return np.round(score, decimal) if one_dim else self.get_multi_output_result(score, multi_output, decimal)

    def overall_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Overall Index (OI): Best possible value = 1, bigger value is better. Range = [-inf, +1]

        Links:
            + https://doi.org/10.1016/j.solener.2019.01.037

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): OI metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        ec = ru.calculate_ec(y_true, y_pred, one_dim)
        rmse = np.sqrt(ru.calculate_mse(y_true, y_pred, one_dim))
        if one_dim:
            score = (1 - rmse / (np.max(y_true) - np.min(y_true)) + ec) / 2.0
            return np.round(score, decimal)
        else:
            score = (1 - rmse / (np.max(y_true, axis=0) - np.min(y_true, axis=0)) + ec) / 2.0
            return self.get_multi_output_result(score, multi_output, decimal)

    def coefficient_of_residual_mass(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Coefficient of Residual Mass (CRM): Best possible value = 0.0, smaller value is better. Range = [-inf, +inf]

        Links:
            + https://doi.org/10.1016/j.csite.2022.101797

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): CRM metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round((np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true), decimal)
        else:
            score = (np.sum(y_pred, axis=0) - np.sum(y_true, axis=0)) / np.sum(y_true, axis=0)
            return self.get_multi_output_result(score, multi_output, decimal)

    def gini_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Gini coefficient (Gini): Best possible score is 1, bigger value is better. Range = [0, 1]

        Notes
        ~~~~~
            + This version is based on below repository matlab code.
            + https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/gini.m

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): Gini metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            idx_sort = np.argsort(-y_pred)
            population_delta = 1.0 / len(y_true)
            accumulated_population_percentage_sum, accumulated_loss_percentage_sum, score = 0, 0, 0
            total_losses = np.sum(y_true)
            for i in range(0, len(y_true)):
                accumulated_loss_percentage_sum += y_true[idx_sort[i]] / total_losses
                accumulated_population_percentage_sum += population_delta
                score += accumulated_loss_percentage_sum - accumulated_population_percentage_sum
            score = score / len(y_true)
            return np.round(score, decimal)
        else:
            col = y_true.shape[1]
            idx_sort = np.argsort(-y_pred, axis=0)
            population_delta = 1.0 / len(y_true)
            accumulated_population_percentage_sum, accumulated_loss_percentage_sum, score = np.zeros(col), np.zeros(col), np.zeros(col)
            total_losses = np.sum(y_true, axis=0)
            for i in range(0, col):
                for j in range(0, len(y_true)):
                    accumulated_loss_percentage_sum[i] += y_true[idx_sort[j, i], i] / total_losses[i]
                    accumulated_population_percentage_sum[i] += population_delta
                    score[i] += accumulated_loss_percentage_sum[i] - accumulated_population_percentage_sum[i]
            result = score / len(y_true)
            return self.get_multi_output_result(result, multi_output, decimal)

    def gini_coefficient_wiki(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Gini coefficient (Gini): Best possible score is 1, bigger value is better. Range = [0, 1]

        Notes
        ~~~~~
            + This version is based on wiki page, may be is the true version
            + https://en.wikipedia.org/wiki/Gini_coefficient
            + Gini coefficient can theoretically range from 0 (complete equality) to 1 (complete inequality)
            + It is sometimes expressed as a percentage ranging between 0 and 100.
            + If negative values are possible, then the Gini coefficient could theoretically be more than 1.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): Gini metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            y = np.concatenate((y_true, y_pred), axis=0)
            score = 0
            for i in range(0, len(y)):
                score += np.sum(np.abs(y[i] - y))
            score = score / (2 * len(y) * np.sum(y))
            return np.round(score, decimal)
        else:
            y = np.concatenate((y_true, y_pred), axis=0)
            col = y.shape[1]
            d = len(y)
            score = np.zeros(col)
            for k in range(0, col):
                for i in range(0, d):
                    for j in range(0, d):
                        score[k] += np.abs(y[i, k] - y[j, k])
            result = score / (2 * len(y) ** 2 * np.mean(y, axis=0))
            return self.get_multi_output_result(result, multi_output, decimal)

    def single_relative_error(self, y_true=None, y_pred=None, decimal=None, non_zero=True, positive=False, **kwargs):
        """
        Relative Error (RE): Best possible score is 0.0, smaller value is better. Range = (-inf, +inf)

        Note: Computes the relative error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (np.ndarray): RE metric
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 0)
        else:
            y_true[y_true == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        return np.round(y_pred / y_true - 1, decimal)

    def single_absolute_error(self, y_true=None, y_pred=None, decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Absolute Error (AE): Best possible score is 0.0, smaller value is better. Range = (-inf, +inf)

        Note: Computes the absolute error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (np.ndarray): AE metric
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        return np.round(np.abs(y_true) - np.abs(y_pred), decimal)

    def single_squared_error(self, y_true=None, y_pred=None, decimal=None, non_zero=False, positive=False, **kwargs):
        """
        Squared Error (SE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Note: Computes the squared error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (np.ndarray): SE metric
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        return np.round((y_true - y_pred) ** 2, decimal)

    def single_squared_log_error(self, y_true=None, y_pred=None, decimal=None, non_zero=True, positive=True, **kwargs):
        """
        Squared Log Error (SLE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Note: Computes the squared log error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (np.ndarray): SLE metric
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = du.get_regression_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = du.get_regression_positive_data(y_true, y_pred, one_dim, 2)
        return np.round((np.log(y_true) - np.log(y_pred)) ** 2, decimal)

    EVS = explained_variance_score
    ME = max_error
    MBE = mean_bias_error
    MAE = mean_absolute_error
    MSE = mean_squared_error
    RMSE = root_mean_squared_error
    MSLE = mean_squared_log_error
    MedAE = median_absolute_error
    MRE = MRB = mean_relative_bias = mean_relative_error
    MPE = mean_percentage_error
    MAPE = mean_absolute_percentage_error
    SMAPE = symmetric_mean_absolute_percentage_error
    MAAPE = mean_arctangent_absolute_percentage_error
    MASE = mean_absolute_scaled_error
    NSE = nash_sutcliffe_efficiency
    NNSE = normalized_nash_sutcliffe_efficiency
    WI = willmott_index
    R = PCC = pearson_correlation_coefficient
    AR = APCC = absolute_pearson_correlation_coefficient
    RSQ = R2S = pearson_correlation_coefficient_square
    CI = confidence_index
    COD = R2 = coefficient_of_determination
    ACOD = AR2 = adjusted_coefficient_of_determination
    DRV = deviation_of_runoff_volume
    KGE = kling_gupta_efficiency
    PCD = prediction_of_change_in_direction
    CE = cross_entropy
    KLD = kullback_leibler_divergence
    JSD = jensen_shannon_divergence
    VAF = variance_accounted_for
    RAE = relative_absolute_error
    A10 = a10_index
    A20 = a20_index
    A30 = a30_index
    NRMSE = normalized_root_mean_square_error
    RSE = residual_standard_error
    COV = covariance
    COR = correlation
    EC = efficiency_coefficient
    OI = overall_index
    CRM = coefficient_of_residual_mass
    GINI = gini_coefficient
    GINI_WIKI = gini_coefficient_wiki

    RE = RB = single_relative_bias = single_relative_error
    AE = single_absolute_error
    SE = single_squared_error
    SLE = single_squared_log_error
