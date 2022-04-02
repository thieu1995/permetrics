#!/usr/bin/env python
# Created by "Thieu" at 10:48, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from copy import deepcopy


class Evaluator:
    """
    This is base class for all performance metrics
    """

    EPSILON = 1e-10

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point
        """
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.y_true_original = deepcopy(y_true)
        self.y_pred_original = deepcopy(y_pred)
        self.y_true = deepcopy(y_true)
        self.y_pred = deepcopy(y_pred)
        self.decimal = decimal
        self.one_dim = False

    def set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __format_data_type(self, y_true, y_pred):
        if isinstance(y_true, (list, tuple, np.ndarray)) and isinstance(y_pred, (list, tuple, np.ndarray)):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            ## Remove all dimensions of size 1
            y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
            if y_true.ndim == y_pred.ndim:
                # x = x[~np.isnan(x)] can't remove if array is dtype object, only work with dtype float
                return y_true.astype('float64'), y_pred.astype('float64')
            else:
                print("Permetrics Error! y_true and y_pred need to have same number of dimensions.")
                exit(0)
        else:
            print("Permetrics Error! y_true and y_pred need to be a list, tuple or np.array.")
            exit(0)

    def __format_data(self, y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.ndim == y_pred.ndim == 1:
            ## Remove all Nan in y_pred
            y_true = y_true[~np.isnan(y_pred)]
            y_pred = y_pred[~np.isnan(y_pred)]
            ## Remove all Inf in y_pred
            y_true = y_true[np.isfinite(y_pred)]
            y_pred = y_pred[np.isfinite(y_pred)]
            return y_true, y_pred, True
        elif y_true.ndim == y_pred.ndim > 1:
            ## Remove all row with Nan in y_pred
            y_true = y_true[~np.isnan(y_pred).any(axis=1)]
            y_pred = y_pred[~np.isnan(y_pred).any(axis=1)]
            ## Remove all row with Inf in y_pred
            y_true = y_true[np.isfinite(y_pred).all(axis=1)]
            y_pred = y_pred[np.isfinite(y_pred).all(axis=1)]
            return y_true, y_pred, False
        
    def get_processed_data(self, y_true=None, y_pred=None, decimal=None):
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
            y_true, y_pred = self.__format_data_type(y_true, y_pred)
            y_true, y_pred, one_dim = self.__format_data(y_true, y_pred)
        else:
            if (self.y_true is not None) and (self.y_pred is not None):
                y_true, y_pred = self.__format_data_type(self.y_true, self.y_pred)
                y_true, y_pred, one_dim = self.__format_data(y_true, y_pred)
            else:
                print("Permetrics Error! You need to pass y_true and y_pred to object creation or function called.")
                exit(0)
        return y_true, y_pred, one_dim, decimal
    
    def get_non_zero_data(self, y_true, y_pred, one_dim=True, rule_idx=0):
        """
        Get non-zero data based on rule

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            one_dim (bool): is y_true has 1 dimensions or not
            rule_idx (int): valid values [0, 1, 2] corresponding to [y_true, y_pred, both true and pred]

        Returns:
            y_true: y_true with positive values based on rule
            y_pred: y_pred with positive values based on rule

        """
        if rule_idx == 0:
            y_rule = deepcopy(y_true)
        elif rule_idx == 1:
            y_rule = deepcopy(y_pred)
        else:
            if one_dim:
                y_true_non, y_pred_non = y_true[y_true != 0], y_pred[y_true != 0]
                y_true, y_pred = y_true_non[y_pred_non != 0], y_pred_non[y_pred_non != 0]
            else:
                y_true_non, y_pred_non = y_true[~np.any(y_true == 0, axis=1)], y_pred[~np.any(y_true == 0, axis=1)]
                y_true, y_pred = y_true_non[~np.any(y_pred_non == 0, axis=1)], y_pred_non[~np.any(y_pred_non == 0, axis=1)]
            return y_true, y_pred
        if one_dim:
            y_true, y_pred = y_true[y_rule != 0], y_pred[y_rule !=0]
        else:
            y_true, y_pred = y_true[~np.any(y_rule == 0, axis=1)], y_pred[~np.any(y_rule == 0, axis=1)]
        return y_true, y_pred
    
    def get_positive_data(self, y_true, y_pred, one_dim=True, rule_idx=0):
        """
        Get positive data based on rule

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            one_dim (bool): is y_true has 1 dimensions or not
            rule_idx (int): valid values [0, 1, 2] corresponding to [y_true, y_pred, both true and pred]

        Returns:
            y_true: y_true with positive values based on rule
            y_pred: y_pred with positive values based on rule
        """
        if rule_idx == 0:
            y_rule = deepcopy(y_true)
        elif rule_idx == 1:
            y_rule = deepcopy(y_pred)
        else:
            if one_dim:
                y_true_non, y_pred_non = y_true[y_true > 0], y_pred[y_true > 0]
                y_true, y_pred = y_true_non[y_pred_non > 0], y_pred_non[y_pred_non > 0]
            else:
                y_true_non, y_pred_non = y_true[np.all(y_true > 0, axis=1)], y_pred[np.all(y_true > 0, axis=1)]
                y_true, y_pred = y_true_non[np.all(y_pred_non > 0, axis=1)], y_pred_non[np.all(y_pred_non > 0, axis=1)]
            return y_true, y_pred
        if one_dim:
            y_true, y_pred = y_true[y_rule > 0], y_pred[y_rule > 0]
        else:
            y_true, y_pred = y_true[np.all(y_rule > 0, axis=1)], y_pred[np.all(y_rule > 0, axis=1)]
        return y_true, y_pred

    def get_multi_output_result(self, result=None, multi_output=None, decimal=None):
        """
        Get multiple output results based on selected parameter

        Args:
            result: The raw result from metric
            multi_output: "raw_values" - return multi-output, [weights] - return single output based on weights, else - return mean result
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            final_result: Multiple outputs results based on selected parameter
        """
        if isinstance(multi_output, (tuple, list, set, np.ndarray)):
            weights = np.array(multi_output)
            if self.y_true.shape[1] != len(weights):
                print("Permetrics Error! Multi-output weights has different length with y_true")
                exit(0)
            return np.round(np.dot(result, multi_output), decimal)
        elif multi_output == "raw_values":  # Default: raw_values
            return np.round(result, decimal)
        else:
            return np.round(np.mean(result), decimal)
