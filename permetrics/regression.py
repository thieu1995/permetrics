# !/usr/bin/env python
# Created by "Thieu" at 18:07, 18/07/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from permetrics.evaluator import Evaluator
from permetrics.utils import *
import numpy as np


class RegressionMetric(Evaluator):
    """
    This is class contains all regression metrics (for both regression and time-series problem)

    Notes
    ~~~~~
    + An extension of scikit-learn metrics section, besides so many new metrics
    + Some methods in scikit-learn can't generate the multi-output metrics, we re-implement all of them and allow multi-output metrics
    + Therefore, this class can calculate the multi-output metrics for all methods
    + https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point
            **kwargs ():
        """
        super().__init__(y_true, y_pred, decimal, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)

    def explained_variance_score(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(1 - np.var(y_true - y_pred) / np.var(y_true), decimal)
        else:
            result = 1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def max_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.max(np.abs(y_true - y_pred)), decimal)
        else:
            result = np.max(np.abs(y_true - y_pred), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(np.abs(y_pred - y_true)) / len(y_true), decimal)
        else:
            result = np.sum(np.abs(y_pred - y_true), axis=0) / len(y_true)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum((y_pred - y_true) ** 2) / len(y_true), decimal)
        else:
            result = np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true)
            return self.get_multi_output_result(result, multi_output, decimal)

    def root_mean_squared_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sqrt(np.sum((y_pred - y_true) ** 2) / len(y_true)), decimal)
        else:
            result = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true))
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_squared_log_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=True):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(np.log((y_true + 1) / (y_pred+1)) ** 2) / len(y_true), decimal)
        else:
            result = np.sum(np.log((y_true + 1) / (y_pred + 1)) ** 2, axis=0) / len(y_true)
            return self.get_multi_output_result(result, multi_output, decimal)

    def median_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.median(np.abs(y_true - y_pred)), decimal)
        else:
            result = np.median(np.abs(y_true - y_pred), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_relative_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False):
        """
        Mean Relative Error (MRE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): MRE metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 0)
        else:
            y_true[y_true == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs((y_true - y_pred) / y_true)), decimal)
        else:
            result = np.mean(np.abs((y_true - y_pred) / y_true), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 0)
        else:
            y_true[y_true == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)), decimal)
        else:
            result = np.mean(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def symmetric_mean_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))), decimal)
        else:
            result = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_arctangent_absolute_percentage_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.arctan(np.abs((y_true - y_pred)/y_true))), decimal)
        else:
            result = np.mean(np.arctan(np.abs((y_true - y_pred)/y_true)), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def mean_absolute_scaled_error(self, y_true=None, y_pred=None, m=1, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[m:] - y_true[:-m])), decimal)
        else:
            result = np.mean(np.abs(y_true - y_pred), axis=0) / np.mean(np.abs(y_true[m:] - y_true[:-m]), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def nash_sutcliffe_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        nse = calculate_nse(y_true, y_pred, one_dim)
        return np.round(nse, decimal) if one_dim else self.get_multi_output_result(nse, multi_output, decimal)

    def normalized_nash_sutcliffe_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        nse = calculate_nse(y_true, y_pred, one_dim)
        nnse = 1 / (2 - nse)
        return np.round(nnse, decimal) if one_dim else self.get_multi_output_result(nnse, multi_output, decimal)

    def willmott_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        wi = calculate_wi(y_true, y_pred, one_dim)
        return np.round(wi, decimal) if one_dim else self.get_multi_output_result(wi, multi_output, decimal)

    def coefficient_of_determination(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = False)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = False)

        Returns:
            result (float, int, np.ndarray): R2 metric for single column or multiple columns
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2), decimal)
        else:
            result = 1 - np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def pearson_correlation_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        pcc = calculate_pcc(y_true, y_pred, one_dim)
        return np.round(pcc, decimal) if one_dim else self.get_multi_output_result(pcc, multi_output, decimal)

    def absolute_pearson_correlation_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        pcc = calculate_absolute_pcc(y_true, y_pred, one_dim)
        return np.round(pcc, decimal) if one_dim else self.get_multi_output_result(pcc, multi_output, decimal)

    def pearson_correlation_coefficient_square(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        pcc = calculate_pcc(y_true, y_pred, one_dim)
        return np.round(pcc**2, decimal) if one_dim else self.get_multi_output_result(pcc**2, multi_output, decimal)

    def confidence_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
        - ≤ 0.40,          Very bad

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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        r = calculate_pcc(y_true, y_pred, one_dim)
        d = calculate_wi(y_true, y_pred, one_dim)
        return np.round(r*d, decimal) if one_dim else self.get_multi_output_result(r*d, multi_output, decimal)

    def deviation_of_runoff_volume(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(y_pred) / np.sum(y_true), decimal)
        else:
            result = np.sum(y_pred, axis=0) / np.sum(y_true, axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def kling_gupta_efficiency(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
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

    def gini_coefficient(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
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

    def gini_coefficient_wiki(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
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

    def prediction_of_change_in_direction(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            d = np.diff(y_true)
            dp = np.diff(y_pred)
            return np.round(np.mean(np.sign(d) == np.sign(dp)), decimal)
        else:
            d = np.diff(y_true, axis=0)
            dp = np.diff(y_pred, axis=0)
            result = np.mean(np.sign(d) == np.sign(dp), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def cross_entropy(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=True):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        entropy = calculate_entropy(y_true, y_pred, one_dim)
        return np.round(entropy, decimal) if one_dim else self.get_multi_output_result(entropy, multi_output, decimal)

    def kullback_leibler_divergence(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=True):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.sum(y_true * np.log2(y_true / y_pred)), decimal)
        else:
            score = np.sum(y_true * np.log2(y_true / y_pred), axis=0)
            return self.get_multi_output_result(score, multi_output, decimal)

    def jensen_shannon_divergence(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=True):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            m = 0.5 * (y_true + y_pred)
            score = 0.5 * np.sum(y_true * np.log2(y_true / m)) + 0.5 * np.sum(y_pred * np.log2(y_pred / m))
            return np.round(score, decimal)
        else:
            m = 0.5 * (y_true + y_pred)
            score = 0.5 * np.sum(y_true * np.log2(y_true / m), axis=0) + 0.5 * np.sum(y_pred * np.log2(y_pred / m), axis=0)
            return self.get_multi_output_result(score, multi_output, decimal)

    def variance_accounted_for(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 0)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)

        if one_dim:
            return np.round((1 - np.var(y_true - y_pred) / np.var(y_true)) * 100, decimal)
        else:
            result = (1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)) * 100
            return self.get_multi_output_result(result, multi_output, decimal)

    def relative_absolute_error(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            numerator = np.power(np.sum((y_pred - y_true)**2), 1/2)
            denominator = np.power(np.sum(y_true**2), 1/2)
            return np.round(numerator/denominator, decimal)
        else:
            numerator = np.power(np.sum((y_pred - y_true) ** 2, axis=0), 1 / 2)
            denominator = np.power(np.sum(y_true ** 2, axis=0), 1 / 2)
            return self.get_multi_output_result(numerator/denominator, multi_output, decimal)

    def a10_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)

        if one_dim:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.9, div <= 1.1), 1, 0)
            return np.round(np.mean(div), decimal)
        else:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.9, div <= 1.1), 1, 0)
            return self.get_multi_output_result(np.mean(div, axis=0), multi_output, decimal)

    def a20_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return np.round(np.mean(div), decimal)
        else:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return self.get_multi_output_result(np.mean(div, axis=0), multi_output, decimal)

    def a30_index(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.7, div <= 1.3), 1, 0)
            return np.round(np.mean(div), decimal)
        else:
            div = y_true / y_pred
            div = np.where(np.logical_and(div >= 0.7, div <= 1.3), 1, 0)
            return self.get_multi_output_result(np.mean(div, axis=0), multi_output, decimal)

    def normalized_root_mean_square_error(self, y_true=None, y_pred=None, model=0, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
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

    def residual_standard_error(self, y_true=None, y_pred=None, n_paras=None, multi_output="raw_values", decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            ss_residuals = np.sum((y_true - y_pred)**2)
            df_residuals = len(y_true) - n_paras - 1
            return np.round(np.sqrt(ss_residuals / df_residuals), decimal)
        else:
            ss_residuals = np.sum((y_true - y_pred) ** 2, axis=0)
            df_residuals = len(y_true) - n_paras - 1
            score = np.sqrt(ss_residuals / df_residuals)
            return self.get_multi_output_result(score, multi_output, decimal)

    def single_relative_error(self, y_true=None, y_pred=None, decimal=None, non_zero=True, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 0)
        else:
            y_true[y_true == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        return np.round(y_pred / y_true - 1, decimal)

    def single_absolute_error(self, y_true=None, y_pred=None, decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        return np.round(np.abs(y_true) - np.abs(y_pred), decimal)

    def single_squared_error(self, y_true=None, y_pred=None, decimal=None, non_zero=False, positive=False):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 2)
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        return np.round((y_true - y_pred) ** 2, decimal)

    def single_squared_log_error(self, y_true=None, y_pred=None, decimal=None, non_zero=True, positive=True):
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
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        if positive:
            y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        return np.round((np.log(y_true) - np.log(y_pred)) ** 2, decimal)

    def get_metric_by_name(self, metric_name=str, paras=None) -> dict:
        """
        Get single metric by name, specific parameter of metric by dictionary

        Args:
            metric_name (str): Select name of metric
            paras (dict): Dictionary of hyper-parameter for that metric

        Returns:
            result (dict): { metric_name: value }
        """
        result = {}
        obj = getattr(self, metric_name)
        result[metric_name] = obj() if paras is None else obj(**paras)
        return result

    def get_metrics_by_list_names(self, list_metric_names=list, list_paras=None) -> dict:
        """
        Get results of list metrics by its name and parameters

        Args:
            list_metric_names (list): e.g, ["RMSE", "MAE", "MAPE"]
            list_paras (list): e.g, [ {"decimal": 5, None}, {"decimal": 4, "multi_output": "raw_values"}, {"decimal":6, "multi_output": [2, 3]} ]

        Returns:
            results (dict): e.g, { "RMSE": 0.25, "MAE": [0.3, 0.6], "MAPE": 0.15 }
        """
        results = {}
        for idx, metric_name in enumerate(list_metric_names):
            obj = getattr(self, metric_name)
            if list_paras is None:
                results[metric_name] = obj()
            else:
                if len(list_metric_names) != len(list_paras):
                    print("Permetrics Error! Different length between list of functions and list of parameters.")
                    exit(0)
                if list_paras[idx] is None:
                    results[metric_name] = obj()
                else:
                    results[metric_name] = obj(**list_paras[idx])
        return results

    def get_metrics_by_dict(self, metrics_dict:dict) -> dict:
        """
        Get results of list metrics by its name and parameters wrapped by dictionary

        For example:
            {"RMSE": { "multi_output": multi_output, "decimal": 4 }, "MAE": { "non_zero": True, "multi_output": multi_output, "decimal": 6}}

        Args:
            metrics_dict (dict): key is metric name and value is dict of parameters

        Returns:
            results (dict): e.g, { "RMSE": 0.3524, "MAE": 0.445263 }
        """
        results = {}
        for metric_name, paras_dict in metrics_dict.items():
            obj = getattr(self, metric_name)
            if paras_dict is None:
                results[metric_name] = obj()
            else:
                results[metric_name] = obj(**paras_dict)     # Unpacking a dictionary and passing it to function
        return results

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
    NNSE = nnse = normalized_nash_sutcliffe_efficiency
    WI = wi = willmott_index
    R = r = PCC = pcc = pearson_correlation_coefficient
    AR = ar = APCC = apcc = absolute_pearson_correlation_coefficient
    R2s = r2s = pearson_correlation_coefficient_square
    CI = ci = confidence_index
    R2 = r2 = coefficient_of_determination
    DRV = drv = deviation_of_runoff_volume
    KGE = kge = kling_gupta_efficiency
    GINI = gini = gini_coefficient
    GINI_WIKI = gini_wiki = gini_coefficient_wiki
    PCD = pcd = prediction_of_change_in_direction
    CE = ce = cross_entropy
    KLD = kld = kullback_leibler_divergence
    JSD = jsd = jensen_shannon_divergence
    VAF = vaf = variance_accounted_for
    RAE = rae = relative_absolute_error
    A10 = a10 = a10_index
    A20 = a20 = a20_index
    A30 = a30 = a30_index
    NRMSE = nrmse = normalized_root_mean_square_error
    RSE = rse = residual_standard_error

    RE = re = single_relative_error
    AE = ae = single_absolute_error
    SE = se = single_squared_error
    SLE = sle = single_squared_log_error
