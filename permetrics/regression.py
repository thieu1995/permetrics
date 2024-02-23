#!/usr/bin/env python
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

    def __init__(self, y_true=None, y_pred=None, **kwargs):
        super().__init__(y_true, y_pred, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)

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

    def get_processed_data(self, y_true=None, y_pred=None, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            n_out: Number of outputs
        """
        if (y_true is not None) and (y_pred is not None):
            y_true, y_pred, n_out = du.format_regression_data_type(y_true, y_pred)
        else:
            if (self.y_true is not None) and (self.y_pred is not None):
                y_true, y_pred, n_out = du.format_regression_data_type(self.y_true, self.y_pred)
            else:
                raise ValueError("y_true or y_pred is None. You need to pass y_true and y_pred to object creation or function called.")
        return y_true, y_pred, n_out

    def explained_variance_score(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0.0, **kwargs):
        """
        Explained Variance Score (EVS). Best possible score is 1.0, greater value is better. Range = (-inf, 1.0]

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): EVS metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = 1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def max_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Max Error (ME): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): ME metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.max(np.abs(y_true - y_pred), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_bias_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Bias Error (MBE): Best possible score is 0.0. Range = (-inf, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MBE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean(y_pred - y_true, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Absolute Error (MAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MAE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.sum(np.abs(y_pred - y_true), axis=0) / len(y_true)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Squared Error (MSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MSE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean((y_true - y_pred) ** 2, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def root_mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Root Mean Squared Error (RMSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): RMSE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_squared_log_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Squared Log Error (MSLE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Link: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error-(msle)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MSLE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.sum(np.log((y_true + 1) / (y_pred + 1)) ** 2, axis=0) / len(y_true)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def median_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Median Absolute Error (MedAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MedAE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.median(np.abs(y_true - y_pred), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_relative_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Relative Error (MRE) - Mean Relative Bias (MRB): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MRE (MRB) metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean(np.abs((y_pred - y_true) / y_true), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Percentage Error (MPE): Best possible score is 0.0. Range = (-inf, +inf)
        Link: https://www.dataquest.io/blog/understanding-regression-error-metrics/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MPE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean((y_true - y_pred) / y_true, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Absolute Percentage Error (MAPE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MAPE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def symmetric_mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Symmetric Mean Absolute Percentage Error (SMAPE): Best possible score is 0.0, smaller value is better. Range = [0, 1]
        If you want percentage then multiply with 100%

        Link: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): SMAPE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_arctangent_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Arctangent Absolute Percentage Error (MAAPE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MAAPE metric for single column or multiple columns (radian values)
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean(np.arctan(np.abs((y_true - y_pred) / y_true)), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def mean_absolute_scaled_error(self, y_true=None, y_pred=None, m=1, multi_output="raw_values", force_finite=True, finite_value=1.0, **kwargs):
        """
        Mean Absolute Scaled Error (MASE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Link: https://en.wikipedia.org/wiki/Mean_absolute_scaled_error

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            m (int): m = 1 for non-seasonal data, m > 1 for seasonal data
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): MASE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.mean(np.abs(y_true - y_pred), axis=0) / np.mean(np.abs(y_true[m:] - y_true[:-m]), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def nash_sutcliffe_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0.0, **kwargs):
        """
        Nash-Sutcliffe Efficiency (NSE): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]
        Link: https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): NSE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = ru.calculate_nse(y_true, y_pred)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def normalized_nash_sutcliffe_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0.0, **kwargs):
        """
        Normalize Nash-Sutcliffe Efficiency (NNSE): Best possible score is 1.0, bigger value is better. Range = [0, 1]
        Link: https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): NSE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        nse = ru.calculate_nse(y_true, y_pred)
        result = 1. / (2. - nse)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def willmott_index(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0.0, **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): WI metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = ru.calculate_wi(y_true, y_pred)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def coefficient_of_determination(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0.0, **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): R2 metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = 1 - np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def adjusted_coefficient_of_determination(self, y_true=None, y_pred=None, X_shape=None,
                                              multi_output="raw_values", force_finite=True, finite_value=0.0, **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): AR2 metric for single column or multiple columns
        """
        if X_shape is None:
            raise ValueError("You need to pass the shape of X_train dataset to calculate Adjusted R2.")
        if len(X_shape) != 2 or X_shape[0] < 4 or X_shape[1] < 1:
            raise ValueError("You need to pass the real shape of X_train dataset to calculate Adjusted R2.")
        dft = X_shape[0] - 1.0
        dfe = X_shape[0] - X_shape[1] - 1.0
        df_final = dft / dfe
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = 1 - df_final * np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def pearson_correlation_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=-1.0, **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): R metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = ru.calculate_pcc(y_true, y_pred)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def absolute_pearson_correlation_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Absolute Pearson’s Correlation Coefficient (APCC or AR): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): AR metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = ru.calculate_absolute_pcc(y_true, y_pred)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def pearson_correlation_coefficient_square(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        (Pearson’s Correlation Index)^2 = R^2 = R2S = RSQ (R square): Best possible score is 1.0, bigger value is better. Range = [0, 1]
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): R2s metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = ru.calculate_pcc(y_true, y_pred)**2
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def confidence_index(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): CI (PI) metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        r = ru.calculate_pcc(y_true, y_pred)
        d = ru.calculate_wi(y_true, y_pred)
        result = r*d
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def deviation_of_runoff_volume(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=10., **kwargs):
        """
        Deviation of Runoff Volume (DRV): Best possible score is 1.0, smaller value is better. Range = [0, +inf)
        Link: https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): DRV metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.sum(y_pred, axis=0) / np.sum(y_true, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def kling_gupta_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Kling-Gupta Efficiency (KGE): Best possible score is 1, bigger value is better. Range = (-inf, 1]
        Link: https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): KGE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        m1, m2 = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
        num_r = np.sum((y_true - m1) * (y_pred - m2), axis=0)
        den_r = np.sqrt(np.sum((y_true - m1) ** 2, axis=0)) * np.sqrt(np.sum((y_pred - m2) ** 2, axis=0))
        r = num_r / den_r
        beta = m2 / m1
        gamma = (np.std(y_pred, axis=0) / m2) / (np.std(y_true, axis=0) / m1)
        result = 1. - np.sqrt((r - 1.) ** 2 + (beta - 1.) ** 2 + (gamma - 1.) ** 2)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def prediction_of_change_in_direction(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Prediction of Change in Direction (PCD): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): PCD metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        d = np.diff(y_true, axis=0)
        dp = np.diff(y_pred, axis=0)
        result = np.mean(np.sign(d) == np.sign(dp), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def cross_entropy(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=-1., **kwargs):
        """
        Cross Entropy (CE): Range = (-inf, 0]. Can't give any comment about this one

        Notes
        ~~~~~
            + Greater value of Entropy, the greater the uncertainty for probability distribution and smaller the value the less the uncertainty
            + https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): CE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = ru.calculate_entropy(y_true, y_pred)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def kullback_leibler_divergence(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=-1., **kwargs):
        """
        Kullback-Leibler Divergence (KLD): Best possible score is 0.0 . Range = (-inf, +inf)
        Link: https://machinelearningmastery.com/divergence-between-probability-distributions/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): KLD metric (bits) for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = np.sum(y_true * np.log2(y_true / y_pred), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def jensen_shannon_divergence(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=1., **kwargs):
        """
        Jensen-Shannon Divergence (JSD): Best possible score is 0.0 (identical), smaller value is better . Range = [0, +inf)
        Link: https://machinelearningmastery.com/divergence-between-probability-distributions/

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): JSD metric (bits) for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        m = 0.5 * (y_true + y_pred)
        result = 0.5 * np.sum(y_true * np.log2(y_true / m), axis=0) + 0.5 * np.sum(y_pred * np.log2(y_pred / m), axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def variance_accounted_for(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Variance Accounted For between 2 signals (VAF): Best possible score is 100% (identical signal), bigger value is better. Range = (-inf, 100%]
        Link: https://www.dcsc.tudelft.nl/~jwvanwingerden/lti/doc/html/vaf.html

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): VAF metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = (1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)) * 100
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def relative_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): RAE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        numerator = np.power(np.sum((y_pred - y_true) ** 2, axis=0), 1 / 2.)
        denominator = np.power(np.sum(y_true ** 2, axis=0), 1 / 2.)
        result = numerator/denominator
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def a10_index(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): A10 metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        div = y_true / y_pred
        div = np.where(np.logical_and(div >= 0.9, div <= 1.1), 1, 0)
        result = np.mean(div, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def a20_index(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): A20 metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        div = y_true / y_pred
        div = np.where(np.logical_and(div >= 0.8, div <= 1.2), 1, 0)
        result = np.mean(div, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def a30_index(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        A30 index (A30): Best possible score is 1.0, bigger value is better. Range = [0, 1]

        Note: a30-index evaluated metric by showing the number of samples that fit the prediction values with a deviation of ±30% compared to experimental values

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): A30 metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        div = y_true / y_pred
        div = np.where(np.logical_and(div >= 0.7, div <= 1.3), 1, 0)
        result = np.mean(div, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def normalized_root_mean_square_error(self, y_true=None, y_pred=None, model=0, multi_output="raw_values", force_finite=True, finite_value=1., **kwargs):
        """
        Normalized Root Mean Square Error (NRMSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Link: https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            model (int): Normalize RMSE by different ways, (Optional, default = 0, valid values = [0, 1, 2, 3]
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): NRMSE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        rmse = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true))
        if model == 1:
            result = rmse / np.mean(y_pred, axis=0)
        elif model == 2:
            result = rmse / (np.max(y_true, axis=0) - np.min(y_true, axis=0))
        elif model == 3:
            result = np.sqrt(np.sum(np.log((y_pred + 1) / (y_true + 1)) ** 2, axis=0) / len(y_true))
        else:
            result = rmse / y_pred.std(axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def residual_standard_error(self, y_true=None, y_pred=None, n_paras=None, multi_output="raw_values", force_finite=True, finite_value=1., **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): RSE metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        ss_residuals = np.sum((y_true - y_pred) ** 2, axis=0)
        df_residuals = len(y_true) - n_paras - 1
        result = np.sqrt(ss_residuals / df_residuals)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def covariance(self, y_true=None, y_pred=None, sample=False, multi_output="raw_values", force_finite=True, finite_value=-10., **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): COV metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        denominator = len(y_true) - 1 if sample else len(y_true)
        s1 = y_true - np.mean(y_true, axis=0)
        s2 = y_pred - np.mean(y_pred, axis=0)
        result = np.sum(s1 * s2, axis=0) / denominator
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def correlation(self, y_true=None, y_pred=None, sample=False, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): COR metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        denominator = len(y_true) - 1 if sample else len(y_true)
        s1 = y_true - np.mean(y_true, axis=0)
        s2 = y_pred - np.mean(y_pred, axis=0)
        cov = np.sum(s1 * s2, axis=0) / denominator
        den = np.std(y_true, axis=0) * np.std(y_pred, axis=0)
        result = cov / den
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def efficiency_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Efficiency Coefficient (EC): Best possible value = 1, bigger value is better. Range = [-inf, +1]

        Links:
            + https://doi.org/10.1016/j.solener.2019.01.037

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): EC metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = ru.calculate_ec(y_true, y_pred)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def overall_index(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Overall Index (OI): Best possible value = 1, bigger value is better. Range = [-inf, +1]

        Links:
            + https://doi.org/10.1016/j.solener.2019.01.037

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): OI metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        ec = ru.calculate_ec(y_true, y_pred)
        rmse = np.sqrt(ru.calculate_mse(y_true, y_pred))
        result = (1 - rmse / (np.max(y_true, axis=0) - np.min(y_true, axis=0)) + ec) / 2.0
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def coefficient_of_residual_mass(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=-1.0, **kwargs):
        """
        Coefficient of Residual Mass (CRM): Best possible value = 0.0, smaller value is better. Range = [-inf, +inf]

        Links:
            + https://doi.org/10.1016/j.csite.2022.101797

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): CRM metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        result = (np.sum(y_pred, axis=0) - np.sum(y_true, axis=0)) / np.sum(y_true, axis=0)
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def gini_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Gini coefficient (GINI): Best possible score is 1, bigger value is better. Range = [0, 1]

        Notes
        ~~~~~
            + This version is based on below repository matlab code.
            + https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/gini.m

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): Gini metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
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
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def gini_coefficient_wiki(self, y_true=None, y_pred=None, multi_output="raw_values", force_finite=True, finite_value=0., **kwargs):
        """
        Gini coefficient (GINI_WIKI): Best possible score is 1, bigger value is better. Range = [0, 1]

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
            force_finite (bool): When result is not finite, it can be NaN or Inf.
                Their result will be replaced by `finite_value` (Optional, default = True)
            finite_value (float): The finite value used to replace Inf or NaN result (Optional, default = 0.0)

        Returns:
            result (float, int, np.ndarray): Gini metric for single column or multiple columns
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        y = np.concatenate((y_true, y_pred), axis=0)
        col = y.shape[1]
        d = len(y)
        score = np.zeros(col)
        for k in range(0, col):
            for i in range(0, d):
                for j in range(0, d):
                    score[k] += np.abs(y[i, k] - y[j, k])
        result = score / (2 * len(y) ** 2 * np.mean(y, axis=0))
        return self.get_output_result(result, n_out, multi_output, force_finite, finite_value=finite_value)

    def single_relative_error(self, y_true=None, y_pred=None, **kwargs):
        """
        Relative Error (RE): Best possible score is 0.0, smaller value is better. Range = (-inf, +inf)
        Note: Computes the relative error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values

        Returns:
            result (np.ndarray): RE metric
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        return y_pred / y_true - 1

    def single_absolute_error(self, y_true=None, y_pred=None, **kwargs):
        """
        Absolute Error (AE): Best possible score is 0.0, smaller value is better. Range = (-inf, +inf)
        Note: Computes the absolute error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values

        Returns:
            result (np.ndarray): AE metric
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        return np.abs(y_true) - np.abs(y_pred)

    def single_squared_error(self, y_true=None, y_pred=None, **kwargs):
        """
        Squared Error (SE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Note: Computes the squared error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values

        Returns:
            result (np.ndarray): SE metric
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        return (y_true - y_pred) ** 2

    def single_squared_log_error(self, y_true=None, y_pred=None, **kwargs):
        """
        Squared Log Error (SLE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
        Note: Computes the squared log error between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values

        Returns:
            result (np.ndarray): SLE metric
        """
        y_true, y_pred, n_out = self.get_processed_data(y_true, y_pred)
        return (np.log(y_true) - np.log(y_pred)) ** 2

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
